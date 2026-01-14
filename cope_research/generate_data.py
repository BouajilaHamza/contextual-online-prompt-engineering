from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SavedSample:
    path: str
    kind: str  # "simple" | "deep"


def _make_simple(rng: random.Random, i: int) -> str:
    title = rng.choice(["Notes", "Summary", "Overview"])
    body = "\n".join(
        rng.choice(
            [
                "Short paragraph of text.",
                "Another simple line.",
                "Minimal structure, easy to parse.",
                "Low nesting depth.",
            ]
        )
        for _ in range(rng.randint(3, 8))
    )
    return f"# {title}\n\n{body}\n"


def _make_deep(rng: random.Random, i: int) -> str:
    items = []
    for a in range(rng.randint(3, 6)):
        items.append(f"- L1 item {i}.{a}")
        for b in range(rng.randint(2, 5)):
            items.append(f"  - L2 item {i}.{a}.{b}")
            for c in range(rng.randint(1, 3)):
                items.append(f"    - L3 item {i}.{a}.{b}.{c}")
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
    return (
        "# Deeply Nested Doc\n\n"
        "## List\n"
        + "\n".join(items)
        + "\n\n"
        "## Code\n```python\n"
        + code
        + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=os.path.join("cope_research", "data"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    saved: list[SavedSample] = []
    for i in range(args.n):
        kind = "simple" if rng.random() < 0.5 else "deep"
        md = _make_simple(rng, i) if kind == "simple" else _make_deep(rng, i)
        fname = f"{i:04d}_{kind}.md"
        path = os.path.join(args.out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        saved.append(SavedSample(path=path, kind=kind))

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": [asdict(s) for s in saved]}, f, indent=2)
        f.write("\n")

    print(manifest_path)


if __name__ == "__main__":
    main()

