from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStrategy:
    """A prompt strategy (bandit arm)."""

    name: str
    system_prompt: str
    # Used only for simulated token-cost penalties / reporting.
    relative_cost: float


PROMPT_STRATEGIES: list[PromptStrategy] = [
    PromptStrategy(
        name="ZeroShot",
        system_prompt=(
            "You are a JSON converter. Convert the following Markdown to JSON. "
            "Return ONLY valid JSON."
        ),
        relative_cost=1.0,
    ),
    PromptStrategy(
        name="SchemaStrict",
        system_prompt=(
            "You are a strict JSON converter. Convert the following Markdown to JSON.\n"
            "Strictly follow the schema constraints:\n"
            "- Use only the allowed keys described by the user.\n"
            "- Do not hallucinate additional keys.\n"
            "- Return ONLY valid JSON."
        ),
        relative_cost=1.5,
    ),
    PromptStrategy(
        name="ChainOfThoughtSyntactic",
        system_prompt=(
            "You are a strict JSON converter. Convert the following Markdown to JSON.\n"
            "Think step-by-step internally. First identify the root object, then process each "
            "section. Explicitly check for trailing commas before closing arrays/objects.\n"
            "Return ONLY valid JSON."
        ),
        relative_cost=2.5,
    ),
    PromptStrategy(
        name="DivideAndConquer",
        system_prompt=(
            "You are a strict JSON converter. The input may be large.\n"
            "Split the document into coherent sections, convert each to JSON, then merge.\n"
            "Return ONLY valid JSON."
        ),
        relative_cost=3.5,
    ),
]


def build_prompt(
    markdown_content: str,
    action_id: int,
    schema_hint: str | None = None,
) -> str:
    """Build a full prompt for an action.

    This keeps the code path shared between mock and real LLM backends.
    """

    strategy = PROMPT_STRATEGIES[action_id]
    parts: list[str] = [f"SYSTEM:\n{strategy.system_prompt}"]
    if schema_hint:
        parts.append(f"\nSCHEMA_HINT:\n{schema_hint.strip()}")
    parts.append(f"\nMARKDOWN:\n{markdown_content.strip()}\n")
    return "\n".join(parts)

