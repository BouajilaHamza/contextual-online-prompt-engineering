from __future__ import annotations

import re

import numpy as np

_RE_CODE_FENCE = re.compile(r"^```")
_RE_LIST_BULLET = re.compile(r"^\s*-\s+")
_RE_LIST_NUMBERED = re.compile(r"^\s*\d+\.\s+")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_feature_dict(markdown_text: str) -> dict[str, float]:
    """Extract a normalized feature dict (all values in [0, 1]).

    Blueprint features:
    - length: len(text) / 10000 (capped at 1.0)
    - depth: max indentation level / 10 (capped at 1.0)
    - code_ratio: chars inside ``` blocks / total chars
    - list_density: list-like lines / total lines
    - special_char_density: count of {, }, " / total chars
    """

    text = markdown_text or ""
    n_chars = max(1, len(text))
    lines = text.splitlines()
    n_lines = max(1, len(lines))

    length = _clamp01(len(text) / 10000.0)

    # Depth: max leading indentation *levels*; tabs count as 4 spaces.
    # We treat 2 spaces as one "level" (common in Markdown lists).
    max_spaces = 0
    for line in lines:
        if not line.strip():
            continue
        expanded = line.replace("\t", " " * 4)
        spaces = len(expanded) - len(expanded.lstrip(" "))
        max_spaces = max(max_spaces, spaces)
    indent_level = max_spaces / 2.0
    depth = _clamp01(indent_level / 10.0)

    # Code ratio: characters within fenced code blocks / total characters.
    in_code = False
    code_chars = 0
    for line in lines:
        if _RE_CODE_FENCE.match(line.strip()):
            in_code = not in_code
            continue
        if in_code:
            code_chars += len(line) + 1  # + newline
    code_ratio = _clamp01(code_chars / n_chars)

    # List density: fraction of list-like lines.
    list_lines = 0
    for line in lines:
        if _RE_LIST_BULLET.match(line) or _RE_LIST_NUMBERED.match(line):
            list_lines += 1
    list_density = _clamp01(list_lines / n_lines)

    # Special char density: { } "
    special = text.count("{") + text.count("}") + text.count('"')
    special_char_density = _clamp01(special / n_chars)

    return {
        "length": float(length),
        "depth": float(depth),
        "code_ratio": float(code_ratio),
        "list_density": float(list_density),
        "special_char_density": float(special_char_density),
    }


def extract_features(markdown_text: str) -> np.ndarray:
    """Extract a normalized feature vector (all values in [0,1]).

    Order: [length, depth, code_ratio, list_density, special_char_density]
    """

    f = extract_feature_dict(markdown_text)
    return np.array(
        [f["length"], f["depth"], f["code_ratio"], f["list_density"], f["special_char_density"]],
        dtype=float,
    )

