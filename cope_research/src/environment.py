from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests

from .prompts import strategy_for_action
from .validator import JsonErrorType, validate_json_output


@dataclass(frozen=True)
class StepResult:
    reward: float
    generated_json: dict[str, Any] | None
    token_cost: int
    parse_ok: bool
    schema_ok: bool
    error_type: str
    error_detail: str | None
    raw_output: str


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        # Remove leading ```lang? and trailing ```
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from LLM output."""
    s = _strip_code_fences(text)

    # Fast path: direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Heuristic: grab outermost {...}
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _merge_json(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Merge src into dst (best-effort, used for divide-and-conquer)."""
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            continue
        a = dst[k]
        if isinstance(a, dict) and isinstance(v, dict):
            dst[k] = _merge_json(a, v)
        elif isinstance(a, list) and isinstance(v, list):
            dst[k] = a + v
        elif a in (None, "", [], {}):
            dst[k] = v
        # else: keep existing
    return dst


class Environment:
    """Environment that turns (markdown, action) into (reward, cost).

    Backends:
    - mock: deterministic-ish simulation to validate online learning logic
    - groq: OpenAI-compatible chat completion endpoint (optional)
    - llama_cpp: local open-source GGUF model via llama.cpp
    """

    def __init__(
        self,
        backend: str = "mock",
        seed: int = 0,
        cost_penalty: float = 0.02,
        groq_model: str | None = None,
        groq_timeout_s: float = 60.0,
        llama_repo_id: str | None = None,
        llama_filename: str | None = None,
        llama_model_path: str | None = None,
        llama_n_ctx: int = 4096,
        llama_n_threads: int | None = None,
        llama_n_gpu_layers: int = 0,
    ):
        self.backend = backend
        self.rng = random.Random(seed)
        self.cost_penalty = float(cost_penalty)
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_timeout_s = float(groq_timeout_s)
        self.llama_repo_id = llama_repo_id or os.getenv(
            "COPE_GGUF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        )
        self.llama_filename = llama_filename or os.getenv(
            "COPE_GGUF_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        self.llama_model_path = llama_model_path or os.getenv("COPE_GGUF_PATH")
        self.llama_n_ctx = int(llama_n_ctx)
        self.llama_n_threads = int(llama_n_threads) if llama_n_threads is not None else None
        self.llama_n_gpu_layers = int(llama_n_gpu_layers)
        self._llm = None

    def step(
        self,
        markdown_content: str,
        action_id: int,
        expected_keys: tuple[str, ...],
        sample_kind: str | None = None,
    ) -> StepResult:
        if self.backend == "mock":
            return self._step_mock(markdown_content, action_id, expected_keys, sample_kind)
        if self.backend == "groq":
            return self._step_groq(markdown_content, action_id, expected_keys)
        if self.backend == "llama_cpp":
            return self._step_llama_cpp(markdown_content, action_id, expected_keys)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _token_cost_for_action(self, action_id: int, markdown_content: str) -> int:
        # very rough proxy: base + proportional to length + per-strategy multiplier
        base = 200
        per_char = 0.02
        rel = strategy_for_action(action_id).relative_cost
        return int((base + per_char * len(markdown_content)) * rel)

    def _apply_reward_shaping(
        self,
        parse_ok: bool,
        schema_ok: bool,
        error_type: JsonErrorType | None,
        token_cost: int,
    ) -> float:
        # Blueprint reward:
        # +1.0 if parses AND schema matches, -1.0 if parse fails, -0.5 if parses but misses keys.
        if not parse_ok:
            base = -1.0
        elif not schema_ok:
            base = -0.5
        else:
            base = 1.0

        # Optional: slightly differentiate syntax error types (still <= 0).
        # This is useful for Phase-2 analysis but doesn't change the sign.
        if (not parse_ok) and error_type in {JsonErrorType.TRAILING_COMMA, JsonErrorType.UNQUOTED_KEY}:
            base = min(base, -0.9)

        # cost penalty discourages over-using expensive arms
        return float(base - self.cost_penalty * (token_cost / 1000.0))

    def _step_mock(
        self,
        markdown_content: str,
        action_id: int,
        expected_keys: tuple[str, ...],
        sample_kind: str | None,
    ) -> StepResult:
        token_cost = self._token_cost_for_action(action_id, markdown_content)

        # Define "optimal" arm mapping by type to make learning measurable:
        # A -> 0, B -> 2, C -> 1. Arm 3 is a fallback for very long inputs (rare here).
        optimal = {"A": 0, "B": 2, "C": 1}.get(sample_kind or "", 2)

        # Probability of producing valid JSON depends on action & kind.
        # Makes arms meaningfully different but not perfectly deterministic.
        if sample_kind == "A":
            p_ok = [0.92, 0.88, 0.85, 0.80][action_id]
        elif sample_kind == "B":
            p_ok = [0.35, 0.55, 0.90, 0.80][action_id]
        elif sample_kind == "C":
            p_ok = [0.50, 0.90, 0.78, 0.70][action_id]
        else:
            p_ok = [0.55, 0.65, 0.80, 0.75][action_id]

        # Add a small advantage when choosing the optimal arm.
        if action_id == optimal:
            p_ok = min(0.97, p_ok + 0.05)

        parse_ok = self.rng.random() < p_ok
        raw_output: str
        if not parse_ok:
            raw_output = '{"oops": "missing brace"'  # invalid JSON
        else:
            generated = {"title": "Doc"}
            # Sometimes schema misses keys when not using schema-like strategies.
            miss_prob = 0.08 if action_id in (1, 2, 3) else 0.18
            if self.rng.random() > miss_prob:
                for k in expected_keys:
                    if k not in generated:
                        generated[k] = [] if k in ("items", "requirements", "code_blocks") else {}
            else:
                if expected_keys:
                    omit = self.rng.choice(list(expected_keys))
                    for k in expected_keys:
                        if k == omit:
                            continue
                        generated[k] = [] if k in ("items", "requirements", "code_blocks") else {}
            raw_output = json.dumps(generated)

        v = validate_json_output(raw_output, expected_keys=expected_keys)
        reward = self._apply_reward_shaping(
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=v.error_type,
            token_cost=token_cost,
        )
        return StepResult(
            reward=reward,
            generated_json=v.json_obj,
            token_cost=token_cost,
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=str(v.error_type),
            error_detail=v.error_detail,
            raw_output=raw_output,
        )

    def _step_groq(
        self,
        markdown_content: str,
        action_id: int,
        expected_keys: tuple[str, ...],
    ) -> StepResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (required for backend=groq).")

        token_cost = self._token_cost_for_action(action_id, markdown_content)
        schema_hint = f"Allowed keys: {list(expected_keys)}"

        # Groq supports OpenAI-compatible chat completions.
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": strategy_for_action(action_id).system_prompt},
                {"role": "user", "content": f"{schema_hint}\n\n{markdown_content}"},
            ],
            "temperature": 0.0,
        }

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=self.groq_timeout_s)
        dt = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        raw_output = data["choices"][0]["message"]["content"]

        v = validate_json_output(raw_output, expected_keys=expected_keys)

        # If usage is provided, use it as a better proxy for cost.
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int) and total_tokens > 0:
            token_cost = total_tokens

        reward = self._apply_reward_shaping(
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=v.error_type,
            token_cost=token_cost,
        )
        # dt currently unused, but could be logged later.
        _ = dt

        return StepResult(
            reward=reward,
            generated_json=v.json_obj if v.parse_ok else None,
            token_cost=token_cost,
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=str(v.error_type),
            error_detail=v.error_detail,
            raw_output=raw_output,
        )

    def _ensure_llama(self):
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it and retry."
            ) from e

        model_path = self.llama_model_path
        if not model_path:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "huggingface-hub is not installed (needed to auto-download GGUF)."
                ) from e
            model_path = hf_hub_download(repo_id=self.llama_repo_id, filename=self.llama_filename)

        # Qwen-instruct GGUFs commonly use ChatML tokens.
        # We embed the ChatML formatting ourselves, so we only need raw completion.
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.llama_n_ctx,
            n_threads=self.llama_n_threads,
            n_gpu_layers=self.llama_n_gpu_layers,
            logits_all=False,
            verbose=False,
        )
        return self._llm

    def _chatml_prompt(self, system: str, user: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{system.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _llama_complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        llm = self._ensure_llama()
        prompt = self._chatml_prompt(system_prompt, user_prompt)

        # llama-cpp returns dict with choices[0].text for completion API.
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>", "</s>"],
        )
        text = (out["choices"][0]["text"] or "").strip()

        # Token accounting (best-effort).
        token_cost = self._token_cost_for_action(0, "")  # fallback base
        try:
            prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
            out_tokens = len(llm.tokenize(text.encode("utf-8")))
            token_cost = prompt_tokens + out_tokens
        except Exception:
            token_cost = self._token_cost_for_action(0, prompt + text)
        return text, int(token_cost)

    def _step_llama_cpp(
        self,
        markdown_content: str,
        action_id: int,
        expected_keys: tuple[str, ...],
    ) -> StepResult:
        schema_hint = f"Allowed keys only: {list(expected_keys)}"
        system_prompt = strategy_for_action(action_id).system_prompt

        if action_id == 3:
            # Divide & conquer: chunk by paragraphs for stability.
            chunks: list[str] = []
            buf: list[str] = []
            buf_len = 0
            for para in (markdown_content or "").split("\n\n"):
                p = para.strip()
                if not p:
                    continue
                if buf_len + len(p) + 2 > 1800 and buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_len = [], 0
                buf.append(p)
                buf_len += len(p) + 2
            if buf:
                chunks.append("\n\n".join(buf))

            merged: dict[str, Any] = {}
            total_cost = 0
            raw_outputs: list[str] = []
            for chunk in chunks:
                user_prompt = f"{schema_hint}\n\nConvert this Markdown chunk to JSON:\n\n{chunk}"
                raw, cost = self._llama_complete(system_prompt, user_prompt, max_tokens=256)
                total_cost += cost
                raw_outputs.append(raw)
                obj = _extract_json_object(raw)
                if isinstance(obj, dict):
                    merged = _merge_json(merged, obj)

            raw = json.dumps(merged) if merged else ""
            v = validate_json_output(raw, expected_keys=expected_keys)
            reward = self._apply_reward_shaping(
                parse_ok=v.parse_ok,
                schema_ok=v.schema_ok,
                error_type=v.error_type,
                token_cost=total_cost,
            )
            return StepResult(
                reward=reward,
                generated_json=v.json_obj if v.parse_ok else None,
                token_cost=total_cost,
                parse_ok=v.parse_ok,
                schema_ok=v.schema_ok,
                error_type=str(v.error_type),
                error_detail=v.error_detail,
                raw_output="\n\n---\n\n".join(raw_outputs),
            )

        max_tokens = 256
        if action_id == 2:
            max_tokens = 384
        user_prompt = f"{schema_hint}\n\nConvert the following Markdown to JSON:\n\n{markdown_content}"
        raw_output, token_cost = self._llama_complete(system_prompt, user_prompt, max_tokens=max_tokens)
        v = validate_json_output(raw_output, expected_keys=expected_keys)
        reward = self._apply_reward_shaping(
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=v.error_type,
            token_cost=token_cost,
        )
        return StepResult(
            reward=reward,
            generated_json=v.json_obj if v.parse_ok else None,
            token_cost=token_cost,
            parse_ok=v.parse_ok,
            schema_ok=v.schema_ok,
            error_type=str(v.error_type),
            error_detail=v.error_detail,
            raw_output=raw_output,
        )


class CopeEnvironment:
    """Blueprint mock simulator to validate LinUCB learning logic quickly."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def simulate_step(self, features: dict[str, float], action_id: int) -> tuple[float, float]:
        """Return (reward, cost) following the blueprint rules."""

        depth = float(features.get("depth", 0.0))
        length = float(features.get("length", 0.0))

        # Action ids:
        # 0: ZERO_SHOT, 1: SCHEMA_STRICT, 2: COT_SYNTAX, 3: DIVIDE_CONQUER
        zero_shot = 0
        cot = 2

        cost = [1.0, 1.5, 3.0, 4.0][action_id]

        if depth > 0.5:
            if action_id == cot:
                return 1.0, cost
            if action_id == zero_shot:
                return -1.0, cost

        if length < 0.2:
            if action_id == zero_shot:
                return 1.0, cost
            if action_id == cot:
                return 0.8, cost

        return float(self.rng.uniform(-0.1, 0.1)), cost

