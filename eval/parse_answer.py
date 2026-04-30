"""Extract a single multiple-choice letter from a generation."""
from __future__ import annotations

import json
import re
from typing import List


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def parse_letter(generation: str, n_options: int) -> str | None:
    """Return one of A..(A+n_options-1) or None."""
    if not generation:
        return None
    valid = {chr(ord("A") + i) for i in range(n_options)}

    # Try JSON first.
    s = generation.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    if "{" in s and "}" in s:
        try:
            j = json.loads(s[s.find("{"): s.rfind("}") + 1])
            ans = str(j.get("answer", "")).strip().upper()[:1]
            if ans in valid:
                return ans
        except Exception:
            pass

    # Fallback: first standalone capital letter that is a valid option.
    for m in _LETTER_RE.finditer(generation.upper()):
        if m.group(1) in valid:
            return m.group(1)
    return None


def candidate_text_for_letter(letter: str, candidates: List[str]) -> str:
    """Map letter back to the option text. Returns '' if invalid."""
    if not letter:
        return ""
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(candidates):
        return candidates[idx]
    return ""
