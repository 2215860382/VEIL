"""Build a memory bank for one video with Qwen3-VL.

Accepts a SampledVideo (pre-decoded, uniformly sampled, pre-cropped frames)
and applies a sliding window: each window of `chunk_size` consecutive frames
is captioned by the VLM, producing one MemoryChunk per window.

This guarantees the same fps / max_frames / resolution as the direct-VQA baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from .sample_frames import SampledVideo, sliding_windows
from .schema import MemoryBank, MemoryChunk

DEFAULT_PROMPT = """You are a careful video analyst.
You are shown {n_frames} frames sampled from timestamps {t0:.0f}s - {t1:.0f}s of the video.

Describe ONLY what is visible in these frames. Do not speculate beyond what you see.
Return a strict JSON object with these keys:
- "visual_caption": 1-2 sentences capturing the visible scene (setting, who, doing what).
- "event_summary":  1 sentence summarizing the key event(s) of this window.
- "objects":        list of salient inanimate objects (concise nouns).
- "persons":        list of people / characters by short descriptor (e.g. "a boy in red").
- "actions":        list of verbs/short phrases describing observed actions.
- "ocr":            any text visible on screen (signs, captions, UI). Empty string if none.

Return ONLY the JSON, no prose, no markdown fences.
"""


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(_as_str(x) for x in v if x is not None and str(x).strip())
    return str(v).strip()


def _as_list(v) -> list:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [_as_str(x) for x in v if x is not None and _as_str(x)]
    s = _as_str(v)
    return [s] if s else []


def _make_memory_text(c: dict, t0: float, t1: float) -> str:
    parts = [f"At {t0:.0f}-{t1:.0f}s:"]
    if c.get("event_summary"):
        parts.append(_as_str(c["event_summary"]))
    if c.get("visual_caption"):
        parts.append(_as_str(c["visual_caption"]))
    if c.get("objects"):
        parts.append("Objects: " + ", ".join(_as_list(c["objects"])))
    if c.get("persons"):
        parts.append("Persons: " + ", ".join(_as_list(c["persons"])))
    if c.get("actions"):
        parts.append("Actions: " + ", ".join(_as_list(c["actions"])))
    if c.get("ocr"):
        parts.append("OCR: " + _as_str(c["ocr"]))
    return " ".join(p for p in parts if p)


def _safe_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]
    try:
        raw = json.loads(s)
        if not isinstance(raw, dict):
            raw = {"visual_caption": text.strip()[:300]}
    except Exception:
        raw = {"visual_caption": text.strip()[:300]}
    return {
        "visual_caption": _as_str(raw.get("visual_caption", "")),
        "event_summary":  _as_str(raw.get("event_summary", "")),
        "objects":        _as_list(raw.get("objects", [])),
        "persons":        _as_list(raw.get("persons", [])),
        "actions":        _as_list(raw.get("actions", [])),
        "ocr":            _as_str(raw.get("ocr", "")),
    }


def build_memory_bank(
    sampled: SampledVideo,
    video_id: str,
    vlm,
    chunk_size: int = 8,
    stride: int = 4,
    prompt_template: str = DEFAULT_PROMPT,
    progress: bool = True,
) -> MemoryBank:
    """Caption each sliding window of `chunk_size` frames and assemble a MemoryBank.

    Args:
        sampled:    Pre-sampled video from sample_frames().
        video_id:   Unique video identifier (used for chunk labels).
        vlm:        VLMClient with chat_with_frames().
        chunk_size: Number of frames per VLM call.
        stride:     Hop size between windows (overlap = chunk_size - stride).
        progress:   Show tqdm bar.
    """
    frames = sampled.frames
    timestamps = sampled.timestamps
    n = len(frames)

    windows = list(sliding_windows(n, chunk_size, stride))
    chunks: List[MemoryChunk] = []
    iterator = tqdm(windows, desc=f"build_memory[{video_id}]") if progress else windows

    for chunk_id, (start, end) in enumerate(iterator):
        win_frames = frames[start:end]
        t0 = timestamps[start]
        t1 = timestamps[end - 1]
        prompt = prompt_template.format(n_frames=len(win_frames), t0=t0, t1=t1)
        raw = vlm.chat_with_frames(win_frames, prompt, max_new_tokens=384)
        parsed = _safe_json(raw)
        chunks.append(
            MemoryChunk(
                video_id=video_id,
                chunk_id=chunk_id,
                start_time=t0,
                end_time=t1,
                visual_caption=parsed.get("visual_caption", ""),
                event_summary=parsed.get("event_summary", ""),
                objects=parsed.get("objects", []) or [],
                persons=parsed.get("persons", []) or [],
                actions=parsed.get("actions", []) or [],
                ocr=parsed.get("ocr", "") or "",
                memory_text=_make_memory_text(parsed, t0, t1),
            )
        )

    return MemoryBank(
        video_id=video_id,
        duration=sampled.duration,
        chunk_size=chunk_size,
        stride=stride,
        fps=sampled.target_fps,
        max_frames=sampled.target_max_frames,
        resolution=sampled.target_resolution,
        chunks=chunks,
    )


def memory_bank_path(cache_dir: str | Path, video_id: str) -> Path:
    return Path(cache_dir) / f"{video_id}.json"
