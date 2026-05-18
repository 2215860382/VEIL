"""Shared JSON helpers — extract a dict from messy LLM output and coerce field types."""
from __future__ import annotations

import json
import re

# Fallback patterns for when JSON is truncated (answer key always appears early)
_ANSWER_RE  = re.compile(r'"answer"\s*:\s*"\(?([A-Za-z])\)?"')
_BOOL_RE    = re.compile(r'"(\w+)"\s*:\s*(true|false)', re.IGNORECASE)
_SUFFICIENT_RE = re.compile(r'"sufficient"\s*:\s*"?(true|false|yes|no|sufficient|insufficient)"?',
                            re.IGNORECASE)
# Extract answer from plain reasoning text when no JSON is present
_CHOICE_RE  = re.compile(
    r'(?:answer|correct\s+(?:answer|option|choice)|choose|select|pick)'
    r'\s*(?:is\s*)?[:\(]?\s*(?:option\s+)?\(?([A-D])\)?',
    re.IGNORECASE,
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def extract_json(text: str) -> dict:
    """Best-effort: pull the first {...} object out of an LLM generation. Returns {} on failure."""
    s = (text or "").strip()
    # Strip <think>...</think> blocks before searching for JSON or regex patterns.
    s_nothink = _THINK_RE.sub("", s).strip()
    if s_nothink:
        s = s_nothink
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            out = json.loads(s[i : j + 1])
            if isinstance(out, dict):
                return out
        except Exception:
            pass

    # JSON is truncated or malformed — rescue key fields via regex.
    # "answer" always appears near the top, so truncation rarely loses it.
    result = {}
    m = _ANSWER_RE.search(s)
    if m:
        result["answer"] = m.group(1).upper()
    for bm in _BOOL_RE.finditer(s):
        result[bm.group(1)] = bm.group(2).lower() == "true"
    sm = _SUFFICIENT_RE.search(s)
    if sm:
        val = sm.group(1).lower()
        result["sufficient"] = val in ("true", "yes", "sufficient")
    # Last resort: extract answer letter from plain reasoning text (no JSON at all)
    if "answer" not in result:
        cm = _CHOICE_RE.search(s)
        if cm:
            result["answer"] = cm.group(1).upper()
    return result


def as_str(v) -> str:
    """str-like → str; list-like → joined str; None/missing → ''."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(as_str(x) for x in v if x is not None and str(x).strip())
    if isinstance(v, bool):
        return str(v).lower()
    return str(v).strip()


def as_list(v) -> list:
    """Anything → list[str]."""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [as_str(x) for x in v if x is not None and as_str(x)]
    s = as_str(v)
    return [s] if s else []


def as_bool(v, default: bool = False) -> bool:
    """Anything → bool; tolerates 'true'/'yes'/1/etc."""
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return default
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "sufficient"):
        return True
    if s in ("false", "no", "n", "0", "insufficient"):
        return False
    return default
