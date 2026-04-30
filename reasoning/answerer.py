"""Answerer — produce final letter-choice given evidence.

Two backends:
  * VLAnswerer: Qwen3-VL, sees evidence text + (optional) frames from selected segments.
  * TextAnswerer: Qwen3-8B, sees only evidence text.
"""
from __future__ import annotations

from typing import List, Sequence

from utils.jsonx import as_list, as_str, extract_json

ANSWERER_SYS = """You answer a multiple-choice question about a long video, using ONLY the retrieved evidence (and optionally the corresponding frames if shown).
Pick exactly one option. Explain briefly which evidence chunks support your choice.

Return ONLY a strict JSON object with these keys:
- "answer":   one letter, exactly one of A/B/C/D (or up to the number of options shown).
- "evidence": list of evidence ids cited from the input (e.g. ["E1", "E3"]).
- "rationale": one sentence justifying the answer, citing the evidence.
"""


def _normalize(raw: dict) -> dict:
    return {
        "answer":    as_str(raw.get("answer", ""))[:1].upper(),
        "evidence":  as_list(raw.get("evidence", [])),
        "rationale": as_str(raw.get("rationale", "")),
    }


def _format_evidence(evidence_texts: List[str]) -> str:
    if not evidence_texts:
        return "(no evidence)"
    return "\n".join(f"[E{i+1}] {t}" for i, t in enumerate(evidence_texts))


def _format_options(candidates: List[str]) -> str:
    return "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


class TextAnswerer:
    """LLM-only answerer: feed evidence text → answer. Used for the auxiliary 'Ours-TextAnswer' setup."""

    def __init__(self, llm):
        self.llm = llm

    def answer(self, question: str, candidates: List[str], evidence_texts: List[str]) -> dict:
        user = (
            f"Question: {question}\n"
            f"Options:\n{_format_options(candidates)}\n\n"
            f"Evidence:\n{_format_evidence(evidence_texts)}\n\n"
            "Return the JSON now."
        )
        messages = [
            {"role": "system", "content": ANSWERER_SYS},
            {"role": "user", "content": user},
        ]
        raw = self.llm.chat(messages, max_new_tokens=256, enable_thinking=False)
        return _normalize(extract_json(raw))


class VLAnswerer:
    """VLM answerer: feed evidence text + (optional) frames from selected chunks → answer."""

    def __init__(self, vlm):
        self.vlm = vlm

    def answer(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        evidence_frames: Sequence[Sequence] = (),
    ) -> dict:
        prompt = (
            f"{ANSWERER_SYS}\n\n"
            f"Question: {question}\n"
            f"Options:\n{_format_options(candidates)}\n\n"
            f"Evidence:\n{_format_evidence(evidence_texts)}\n\n"
            "Return the JSON now."
        )
        flat_frames = []
        for fl in evidence_frames:
            flat_frames.extend(fl)
        if flat_frames:
            raw = self.vlm.chat_with_frames(flat_frames, prompt, max_new_tokens=256)
        else:
            # No frames provided — degrade gracefully to text-only via VLM.
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            raw = self.vlm._generate(messages, max_new_tokens=256)
        return _normalize(extract_json(raw))
