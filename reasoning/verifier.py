"""Verifier — judge whether retrieved evidence is sufficient to answer.

If not, propose a new query that targets the missing evidence.
"""
from __future__ import annotations

from typing import List

from utils.jsonx import as_bool, as_str, extract_json

VERIFIER_SYS = """You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.

Be conservative: if the evidence does not directly support a single best option, mark it INSUFFICIENT.
When insufficient, propose ONE concise next search query (<= 20 words) targeting the missing evidence.

Return ONLY a strict JSON object with these keys:
- "is_sufficient": boolean.
- "missing_evidence": one sentence describing what is still missing (empty string if sufficient).
- "next_query": next search query (empty string if sufficient).
- "reason": one sentence justifying the decision.
"""


def _format_evidence(evidence_texts: List[str]) -> str:
    if not evidence_texts:
        return "(no evidence retrieved yet)"
    return "\n".join(f"[E{i+1}] {t}" for i, t in enumerate(evidence_texts))


def _normalize(raw: dict) -> dict:
    return {
        "is_sufficient":    as_bool(raw.get("is_sufficient", False), default=False),
        "missing_evidence": as_str(raw.get("missing_evidence", "")),
        "next_query":       as_str(raw.get("next_query", "")),
        "reason":           as_str(raw.get("reason", "")),
    }


class Verifier:
    def __init__(self, llm):
        self.llm = llm

    def verify(self, question: str, candidates: List[str], evidence_texts: List[str]) -> dict:
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        ev = _format_evidence(evidence_texts)
        user = (
            f"Question: {question}\n"
            f"Options:\n{opts}\n\n"
            f"Retrieved evidence:\n{ev}\n\n"
            "Return the JSON now."
        )
        messages = [
            {"role": "system", "content": VERIFIER_SYS},
            {"role": "user", "content": user},
        ]
        raw = self.llm.chat(messages, max_new_tokens=256, enable_thinking=False)
        return _normalize(extract_json(raw))
