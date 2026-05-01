"""Planner — generate a retrieval query from question + optional missing-evidence analysis.

First call  (missing_analysis=None) : generate initial query AND rubric.
Repair call (missing_analysis given): generate a focused repair query targeting the gap.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from utils.jsonx import as_str, extract_json

PLANNER_SYS = """You generate targeted video search queries for evidence retrieval.

You receive a question, answer options, and — on repair calls — a Missing Evidence Analysis
describing the type and description of evidence that has not yet been found.

## First call (no missing evidence)
Generate:
  - "query":  a concise search query (≤ 20 words) to find the most relevant video segments.
  - "rubric": one sentence stating the concrete criteria that define sufficient evidence
              for this specific question (e.g. "Evidence must show the exact count of red
              objects in the final scene").

## Repair call (missing evidence provided)
Generate:
  - "query":  a NEW query (≤ 20 words) that directly targets the missing evidence type
              and description. Avoid repeating the previous query.
  - "rubric": empty string — the original rubric is reused.

Return ONLY a strict JSON object, no prose, no markdown fences:
{"query": "...", "rubric": "..."}
"""


class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(
        self,
        question: str,
        candidates: List[str],
        missing_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Generate a retrieval query.

        Args:
            question:         The question text.
            candidates:       Answer option strings.
            missing_analysis: None on first call; dict with "type" and "description"
                              on repair calls (from Verifier output).

        Returns:
            {"query": str, "rubric": str}
        """
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))

        if missing_analysis:
            gap_type = missing_analysis.get("type", "")
            gap_desc = missing_analysis.get("description", "")
            missing_block = (
                f"\nMissing Evidence Analysis:\n"
                f"  type: {gap_type}\n"
                f"  description: {gap_desc}\n"
            )
        else:
            missing_block = ""

        user = (
            f"Question: {question}\n"
            f"Options:\n{opts}"
            f"{missing_block}\n"
            "Return the JSON now."
        )
        messages = [
            {"role": "system", "content": PLANNER_SYS},
            {"role": "user",   "content": user},
        ]
        raw = self.llm.chat(messages, max_new_tokens=128, enable_thinking=False)
        parsed = extract_json(raw)
        return {
            "query":  as_str(parsed.get("query",  "")) or question,
            "rubric": as_str(parsed.get("rubric", "")),
        }
