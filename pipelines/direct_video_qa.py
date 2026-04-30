"""Baseline 1: feed the full video + question directly to Qwen3-VL.

Knobs: fps, max_pixels — tweak in configs/*.yaml under `direct_video_qa`.
"""
from __future__ import annotations

from typing import List


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
    video_path: str,
    question: str,
    candidates: List[str],
    vlm,
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
    max_new_tokens: int = 192,
) -> dict:
    prompt = PROMPT_TEMPLATE.format(question=question, options=_format_options(candidates))
    raw = vlm.chat_with_video(video_path, prompt, max_new_tokens=max_new_tokens, fps=fps, max_pixels=max_pixels)
    return {"raw": raw, "evidence_texts": [], "evidence_chunk_ids": []}
