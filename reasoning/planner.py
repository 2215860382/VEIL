"""Planner — turn (question, options) into an initial retrieval query."""
from __future__ import annotations

from typing import List

from utils.jsonx import as_list, as_str, extract_json

PLANNER_SYS = """You design retrieval queries for video evidence search.
Given a question and its multiple-choice options, write a SHORT search query that would retrieve
the most relevant video segments to answer it. Also describe what kind of evidence would settle the question.

Return ONLY a strict JSON object with these keys:
- "search_query": one concise English query (<= 20 words), no quotes inside.
- "target_evidence": one sentence describing what evidence in the video would answer the question.
- "expected_video_clues": list of 2-4 short clue phrases (visible objects, actions, scenes, OCR text).
"""


def _normalize_plan(raw: dict) -> dict:
    return {
        "search_query":         as_str(raw.get("search_query", "")),
        "target_evidence":      as_str(raw.get("target_evidence", "")),
        "expected_video_clues": as_list(raw.get("expected_video_clues", [])),
    }


class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, question: str, candidates: List[str]) -> dict:
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        user = f"Question: {question}\nOptions:\n{opts}\n\nReturn the JSON now."
        messages = [
            {"role": "system", "content": PLANNER_SYS},
            {"role": "user", "content": user},
        ]
        raw = self.llm.chat(messages, max_new_tokens=256, enable_thinking=False)
        plan = _normalize_plan(extract_json(raw))
        if not plan["search_query"]:
            plan["search_query"] = question  # fallback
        return plan
