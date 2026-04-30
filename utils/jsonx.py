"""Shared JSON helpers — extract a dict from messy LLM output and coerce field types."""
from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    """Best-effort: pull the first {...} object out of an LLM generation. Returns {} on failure."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


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
