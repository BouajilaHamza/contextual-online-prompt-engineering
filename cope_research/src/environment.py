from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests

from .prompts import PROMPT_STRATEGIES, build_prompt


@dataclass(frozen=True)
class StepResult:
    reward: float
    generated_json: dict[str, Any] | None
    token_cost: int
    parse_ok: bool
    schema_ok: bool
    raw_output: str


def _validate_schema_keys(obj: dict[str, Any], expected_keys: tuple[str, ...]) -> bool:
    return all(k in obj for k in expected_keys)


class Environment:
    """Environment that turns (markdown, action) into (reward, cost).

    Backends:
    - mock: deterministic-ish simulation to validate online learning logic
    - groq: OpenAI-compatible chat completion endpoint (optional)
    """

    def __init__(
        self,
        backend: str = "mock",
        seed: int = 0,
        cost_penalty: float = 0.02,
        groq_model: str | None = None,
        groq_timeout_s: float = 60.0,
    ):
        self.backend = backend
        self.rng = random.Random(seed)
        self.cost_penalty = float(cost_penalty)
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_timeout_s = float(groq_timeout_s)

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
        raise ValueError(f"Unknown backend: {self.backend}")

    def _token_cost_for_action(self, action_id: int, markdown_content: str) -> int:
        # very rough proxy: base + proportional to length + per-strategy multiplier
        base = 200
        per_char = 0.02
        rel = PROMPT_STRATEGIES[action_id].relative_cost
        return int((base + per_char * len(markdown_content)) * rel)

    def _apply_reward_shaping(
        self,
        parse_ok: bool,
        schema_ok: bool,
        token_cost: int,
    ) -> float:
        if not parse_ok:
            base = -1.0
        elif not schema_ok:
            base = -0.5
        else:
            base = 1.0
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
        schema_ok = False
        generated: dict[str, Any] | None = None
        raw_output: str

        if not parse_ok:
            raw_output = '{"oops": "missing brace"'  # invalid JSON
        else:
            # Sometimes schema misses keys when not using schema-like strategies.
            miss_prob = 0.08 if action_id in (1, 2, 3) else 0.18
            schema_ok = self.rng.random() > miss_prob
            generated = {"title": "Doc"}
            if schema_ok:
                for k in expected_keys:
                    if k not in generated:
                        generated[k] = [] if k in ("items", "requirements", "code_blocks") else {}
            else:
                # omit one expected key if possible
                if expected_keys:
                    omit = self.rng.choice(list(expected_keys))
                    for k in expected_keys:
                        if k == omit:
                            continue
                        generated[k] = [] if k in ("items", "requirements", "code_blocks") else {}
            raw_output = json.dumps(generated)

        reward = self._apply_reward_shaping(parse_ok=parse_ok, schema_ok=schema_ok, token_cost=token_cost)
        return StepResult(
            reward=reward,
            generated_json=generated,
            token_cost=token_cost,
            parse_ok=parse_ok,
            schema_ok=schema_ok,
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
        full_prompt = build_prompt(markdown_content, action_id=action_id, schema_hint=schema_hint)

        # Groq supports OpenAI-compatible chat completions.
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": PROMPT_STRATEGIES[action_id].system_prompt},
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

        parse_ok = False
        schema_ok = False
        generated: dict[str, Any] | None = None
        try:
            generated = json.loads(raw_output)
            parse_ok = isinstance(generated, dict)
            schema_ok = parse_ok and _validate_schema_keys(generated, expected_keys)
        except Exception:
            parse_ok = False
            schema_ok = False

        # If usage is provided, use it as a better proxy for cost.
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int) and total_tokens > 0:
            token_cost = total_tokens

        reward = self._apply_reward_shaping(parse_ok=parse_ok, schema_ok=schema_ok, token_cost=token_cost)
        # dt currently unused, but could be logged later.
        _ = full_prompt, dt

        return StepResult(
            reward=reward,
            generated_json=generated if parse_ok else None,
            token_cost=token_cost,
            parse_ok=parse_ok,
            schema_ok=schema_ok,
            raw_output=raw_output,
        )

