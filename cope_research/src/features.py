from __future__ import annotations

import math
import re

import numpy as np

_RE_CODE_FENCE = re.compile(r"^```")
_RE_LIST_BULLET = re.compile(r"^\s*-\s+")
_RE_LIST_NUMBERED = re.compile(r"^\s*\d+\.\s+")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_features(markdown_text: str) -> np.ndarray:
    """Extract a normalized feature vector from Markdown.

    Features (all in [0,1], with a leading bias term 1.0):
    - feat_length: log-scaled length indicator
    - feat_depth: max indentation depth indicator
    - feat_code_ratio: fraction of lines inside fenced code blocks
    - feat_list_density: fraction of list-like lines
    - feat_special_chars: density of quote/backslash/brace characters
    """

    text = markdown_text or ""
    lines = text.splitlines()
    n_lines = max(1, len(lines))

    # Length: log-scaled by characters, capped.
    n_chars = len(text)
    feat_length = _clamp01(math.log1p(n_chars) / math.log1p(20_000))

    # Depth: measure max leading spaces (tabs count as 4).
    max_indent = 0
    for line in lines:
        if not line.strip():
            continue
        expanded = line.replace("\t", " " * 4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        max_indent = max(max_indent, indent)
    # 0..32 spaces treated as 0..1
    feat_depth = _clamp01(max_indent / 32.0)

    # Code ratio: fraction of lines inside fenced blocks.
    in_code = False
    code_lines = 0
    for line in lines:
        if _RE_CODE_FENCE.match(line.strip()):
            in_code = not in_code
            continue
        if in_code:
            code_lines += 1
    feat_code_ratio = _clamp01(code_lines / n_lines)

    # List density: bullet or numbered list lines.
    list_lines = 0
    for line in lines:
        if _RE_LIST_BULLET.match(line) or _RE_LIST_NUMBERED.match(line):
            list_lines += 1
    feat_list_density = _clamp01(list_lines / n_lines)

    # Special chars density: quotes, backslashes, braces.
    special_chars = sum(text.count(ch) for ch in ['"', "\\", "{", "}"])
    feat_special_chars = _clamp01(special_chars / max(1, n_chars) * 5.0)

    return np.array(
        [
            1.0,  # bias
            feat_length,
            feat_depth,
            feat_code_ratio,
            feat_list_density,
            feat_special_chars,
        ],
        dtype=float,
    )

