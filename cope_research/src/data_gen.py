from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MarkdownSample:
    sample_id: str
    md: str
    kind: str  # "A" | "B" | "C"
    expected_keys: tuple[str, ...]


def _type_a(i: int, rng: random.Random) -> MarkdownSample:
    title = rng.choice(["Notes", "Summary", "Changelog", "Overview"])
    body = "\n".join(
        rng.choice(
            [
                "This is a short paragraph.",
                "Key idea: keep it simple.",
                "No deep nesting here.",
                "Plain text with light structure.",
            ]
        )
        for _ in range(rng.randint(3, 8))
    )
    md = f"# {title}\n\n{body}\n"
    return MarkdownSample(f"A-{i}", md, "A", expected_keys=("title", "text"))


def _type_b(i: int, rng: random.Random) -> MarkdownSample:
    # Deep nesting + code fences.
    items = []
    for k in range(rng.randint(3, 6)):
        items.append(f"- Level 1 item {k}")
        for j in range(rng.randint(2, 5)):
            items.append(f"  - Level 2 item {k}.{j}")
            for t in range(rng.randint(1, 3)):
                items.append(f"    - Level 3 item {k}.{j}.{t}")

    code = "\n".join(
        [
            "def f(x):",
            "    return x + 1",
            "",
            "class C:",
            "    def g(self):",
            "        return {'a': 1, 'b': 2}",
        ]
    )
    md = (
        "# Deep Spec\n\n"
        "## Tasks\n"
        + "\n".join(items)
        + "\n\n"
        "## Example Code\n"
        "```python\n"
        + code
        + "\n```\n"
    )
    return MarkdownSample(f"B-{i}", md, "B", expected_keys=("title", "items", "code_blocks"))


def _type_c(i: int, rng: random.Random) -> MarkdownSample:
    # Schema-like rigid fields.
    name = rng.choice(["Project X", "Dataset Y", "Model Z"])
    owner = rng.choice(["Alice", "Bob", "Carol", "Dave"])
    version = f"{rng.randint(0,2)}.{rng.randint(0,9)}.{rng.randint(0,9)}"
    md = (
        f"# {name}\n\n"
        "## Metadata\n"
        f"- Owner: {owner}\n"
        f"- Version: {version}\n"
        f"- License: MIT\n\n"
        "## Requirements\n"
        "1. Must be reproducible\n"
        "2. Must validate schema keys\n"
        "3. Must avoid hallucinated fields\n"
    )
    return MarkdownSample(
        f"C-{i}",
        md,
        "C",
        expected_keys=("title", "metadata", "requirements"),
    )


def generate_dataset(
    n_a: int = 33,
    n_b: int = 33,
    n_c: int = 34,
    seed: int = 0,
    shuffle: bool = True,
) -> list[MarkdownSample]:
    rng = random.Random(seed)
    samples: list[MarkdownSample] = []
    for i in range(n_a):
        samples.append(_type_a(i, rng))
    for i in range(n_b):
        samples.append(_type_b(i, rng))
    for i in range(n_c):
        samples.append(_type_c(i, rng))
    if shuffle:
        rng.shuffle(samples)
    return samples


def generate_drift_dataset(
    episodes: int = 100,
    split_at: int = 50,
    seed: int = 0,
) -> list[MarkdownSample]:
    """Episodes [0, split_at) are Type A, then switch to Type B."""
    rng = random.Random(seed)
    samples: list[MarkdownSample] = []
    for i in range(episodes):
        if i < split_at:
            samples.append(_type_a(i, rng))
        else:
            samples.append(_type_b(i, rng))
    return samples

