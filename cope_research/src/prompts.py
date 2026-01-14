from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStrategy:
    """A prompt strategy (bandit arm)."""

    name: str
    system_prompt: str
    # Used only for simulated token-cost penalties / reporting.
    relative_cost: float


PROMPT_STRATEGIES: dict[str, str] = {
    "ZERO_SHOT": (
        "You are a helpful data assistant. Convert the provided Markdown content into valid JSON."
    ),
    "SCHEMA_STRICT": (
        "You are a strict syntax validator. Convert Markdown to JSON. Ensure all keys are quoted. "
        "Do not leave trailing commas. Follow the schema exactly."
    ),
    "COT_SYNTAX": (
        "Think step-by-step. First, identify the structure. Second, escape all special characters. "
        "Third, construct the JSON. Verify closing brackets before outputting."
    ),
    "DIVIDE_CONQUER": (
        "Process this file section by section. Handle the header first, then the body. "
        "Merge them into a single JSON object at the end."
    ),
}

# Canonical arm order used everywhere (action_id -> key).
ARM_KEYS: list[str] = ["ZERO_SHOT", "SCHEMA_STRICT", "COT_SYNTAX", "DIVIDE_CONQUER"]

# Relative cost proxy for token economics (used for reporting/penalties).
ARM_RELATIVE_COST: dict[str, float] = {
    "ZERO_SHOT": 1.0,
    "SCHEMA_STRICT": 1.5,
    "COT_SYNTAX": 2.5,
    "DIVIDE_CONQUER": 3.5,
}


def strategy_for_action(action_id: int) -> PromptStrategy:
    key = ARM_KEYS[action_id]
    return PromptStrategy(name=key, system_prompt=PROMPT_STRATEGIES[key], relative_cost=ARM_RELATIVE_COST[key])


def build_prompt(
    markdown_content: str,
    action_id: int,
    schema_hint: str | None = None,
) -> str:
    """Build a full prompt for an action.

    This keeps the code path shared between mock and real LLM backends.
    """

    strategy = strategy_for_action(action_id)
    parts: list[str] = [f"SYSTEM:\n{strategy.system_prompt}"]
    if schema_hint:
        parts.append(f"\nSCHEMA_HINT:\n{schema_hint.strip()}")
    parts.append(f"\nMARKDOWN:\n{markdown_content.strip()}\n")
    return "\n".join(parts)

