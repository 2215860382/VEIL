"""Baseline 1: feed all pre-sampled frames + question directly to Qwen3-VL.

Frames are supplied by the caller (pre-sampled via memory.sample_frames.sample_frames),
guaranteeing the same fps / max_frames / resolution as the memory-bank pipeline.
"""
from __future__ import annotations

from typing import List, Sequence


PROMPT_TEMPLATE = """You are watching a long video.

Question: {question}
Options:
{options}

Pick exactly one option. Return ONLY a strict JSON object:
{{"answer": "<one letter>", "rationale": "<one sentence>"}}
"""


def _format_options(candidates: List[str]) -> str:
    return "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


def run_direct_video_qa(
    frames: Sequence,
    question: str,
    candidates: List[str],
    vlm,
    max_new_tokens: int = 192,
) -> dict:
    """Answer a multiple-choice question by showing all frames to the VLM.

    Args:
        frames:         Pre-sampled frames (np.ndarray or PIL), already at target resolution.
        question:       Question text.
        candidates:     List of answer option strings.
        vlm:            VLMClient instance.
        max_new_tokens: Generation budget.
    """
    prompt = PROMPT_TEMPLATE.format(question=question, options=_format_options(candidates))
    raw = vlm.chat_with_frames(frames, prompt, max_new_tokens=max_new_tokens)
    return {"raw": raw, "evidence_texts": [], "evidence_chunk_ids": []}
