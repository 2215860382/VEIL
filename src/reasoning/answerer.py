"""Answerer — produce final letter-choice given evidence."""
from __future__ import annotations

import re
from typing import List


def _format_evidence(evidence_texts: List[str], offset: int = 0) -> str:
    if not evidence_texts:
        return "(no evidence)"
    return "\n".join(f"--- Segment {i+1+offset} ---\n{t}" for i, t in enumerate(evidence_texts))


def _format_options(candidates: List[str]) -> str:
    return " ".join(f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


_PROMPT = (
    "You are answering a multiple-choice question about a long video.\n"
    "Below are {n} relevant video segments retrieved from the video.\n\n"
    "{evidence}\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter (A/B/C/D) only, no explanation:"
)


class Answerer:
    def __init__(self, model):
        self.model = model

    def answer(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        keyframe_images=(),
        max_evidence_chars: int = 80000,
        focused_texts: List[str] = (),
    ) -> dict:
        all_texts = list(focused_texts) + list(evidence_texts)
        if all_texts and max_evidence_chars:
            per = max_evidence_chars // len(all_texts)
            focused_texts  = [t[:per] for t in focused_texts]
            evidence_texts = [t[:per] for t in evidence_texts]

        evidence = _format_evidence(list(focused_texts) + list(evidence_texts))
        choices  = _format_options(candidates)
        prompt   = _PROMPT.format(
            n=len(focused_texts) + len(evidence_texts),
            evidence=evidence,
            question=question,
            choices=choices,
        )
        frames = [img for img in keyframe_images if img is not None]
        raw = self.model.chat_with_frames(frames, prompt, max_new_tokens=16, enable_thinking=False)
        m = re.search(r"\b([A-D])\b", raw)
        letter = m.group(1) if m else ""
        return {"answer": letter, "evidence": [], "rationale": ""}
