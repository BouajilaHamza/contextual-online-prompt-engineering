from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class JsonErrorType(StrEnum):
    NONE = "none"
    NOT_JSON_OBJECT = "not_json_object"
    TRAILING_COMMA = "trailing_comma"
    UNQUOTED_KEY = "unquoted_key"
    INVALID_ESCAPE = "invalid_escape"
    OTHER_SYNTAX = "other_syntax"
    MISSING_KEYS = "missing_keys"
    EXTRA_KEYS = "extra_keys"


@dataclass(frozen=True)
class ValidationResult:
    parse_ok: bool
    schema_ok: bool
    error_type: JsonErrorType
    error_detail: str | None
    json_obj: dict[str, Any] | None
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]


_RE_FENCE = re.compile(r"^\s*```")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_RE_INVALID_ESCAPE = re.compile(r"\\(?![\"\\/bfnrtu])")
# Detect unquoted keys after { or , :   { key: ... }  or  , key: ...
_RE_UNQUOTED_KEY = re.compile(r'(?s)(?:\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:')


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```") or _RE_FENCE.match(s):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def _extract_candidate_json(text: str) -> str:
    """Return a best-effort JSON candidate string (object/array) from output."""
    s = _strip_code_fences(text)

    # Direct: if it already looks like JSON
    st = s.lstrip()
    if st.startswith("{") or st.startswith("["):
        return s.strip()

    # Heuristic: grab outermost {...} if present
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        return s[start : end + 1].strip()

    # Heuristic: grab outermost [...] if present
    start = s.find("[")
    end = s.rfind("]")
    if 0 <= start < end:
        return s[start : end + 1].strip()

    return s.strip()


def validate_json_output(
    raw_output: str,
    *,
    expected_keys: tuple[str, ...] = (),
    allowed_keys: tuple[str, ...] | None = None,
) -> ValidationResult:
    """Validate and classify an LLM JSON output.

    - **Syntax**: parse errors are categorized (trailing comma, unquoted keys, invalid escape, other).
    - **Semantics**: missing expected keys (and optionally extra keys) are categorized.
    """

    candidate = _extract_candidate_json(raw_output)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Taxonomy heuristics (best-effort).
        if _RE_TRAILING_COMMA.search(candidate):
            et = JsonErrorType.TRAILING_COMMA
        elif _RE_INVALID_ESCAPE.search(candidate):
            et = JsonErrorType.INVALID_ESCAPE
        elif _RE_UNQUOTED_KEY.search(candidate):
            et = JsonErrorType.UNQUOTED_KEY
        else:
            et = JsonErrorType.OTHER_SYNTAX
        return ValidationResult(
            parse_ok=False,
            schema_ok=False,
            error_type=et,
            error_detail=f"{e.msg} (line {e.lineno}, col {e.colno})",
            json_obj=None,
            missing_keys=(),
            extra_keys=(),
        )
    except Exception as e:
        return ValidationResult(
            parse_ok=False,
            schema_ok=False,
            error_type=JsonErrorType.OTHER_SYNTAX,
            error_detail=str(e),
            json_obj=None,
            missing_keys=(),
            extra_keys=(),
        )

    if not isinstance(parsed, dict):
        return ValidationResult(
            parse_ok=True,
            schema_ok=False,
            error_type=JsonErrorType.NOT_JSON_OBJECT,
            error_detail=f"Top-level type was {type(parsed).__name__}, expected object/dict.",
            json_obj=None,
            missing_keys=(),
            extra_keys=(),
        )

    obj: dict[str, Any] = parsed
    missing = tuple(k for k in expected_keys if k not in obj)

    extra: tuple[str, ...] = ()
    if allowed_keys is not None:
        extra = tuple(k for k in obj.keys() if k not in allowed_keys)

    if missing:
        return ValidationResult(
            parse_ok=True,
            schema_ok=False,
            error_type=JsonErrorType.MISSING_KEYS,
            error_detail=f"Missing keys: {list(missing)}",
            json_obj=obj,
            missing_keys=missing,
            extra_keys=extra,
        )

    if extra:
        return ValidationResult(
            parse_ok=True,
            schema_ok=False,
            error_type=JsonErrorType.EXTRA_KEYS,
            error_detail=f"Extra keys: {list(extra)}",
            json_obj=obj,
            missing_keys=missing,
            extra_keys=extra,
        )

    return ValidationResult(
        parse_ok=True,
        schema_ok=True,
        error_type=JsonErrorType.NONE,
        error_detail=None,
        json_obj=obj,
        missing_keys=missing,
        extra_keys=extra,
    )

