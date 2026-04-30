"""Build a memory bank for one video with Qwen3-VL.

For each VideoSegment we ask the VLM to emit a compact, retrieval-oriented description:
visual caption + main events + objects/persons/actions + OCR. Then collapse them into
`memory_text` (the canonical retrieval target).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from .schema import MemoryBank, MemoryChunk
from .segment_video import VideoSegment, segment_video, video_duration

DEFAULT_PROMPT = """You are a careful video segment analyst.
You are shown {n_frames} frames sampled from a {seg_seconds:.0f}-second video segment (timestamp {t0:.0f}s - {t1:.0f}s of the source video).

Describe ONLY what is visible in these frames. Do not speculate beyond what you see.
Return a strict JSON object with these keys:
- "visual_caption": 1-2 sentences capturing the visible scene (setting, who, doing what).
- "event_summary":  1 sentence summarizing the key event(s) of this segment.
- "objects":        list of salient inanimate objects (concise nouns).
- "persons":        list of people / characters by short descriptor (e.g. "a boy in red").
- "actions":        list of verbs/short phrases describing observed actions.
- "ocr":            any text visible on screen (signs, captions, UI). Empty string if none.

Return ONLY the JSON, no prose, no markdown fences.
"""


def _as_str(v) -> str:
    """Coerce arbitrary JSON value to a string. Lists are joined with '; '."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(_as_str(x) for x in v if x is not None and str(x).strip())
    return str(v).strip()


def _as_list(v) -> list:
    """Coerce arbitrary JSON value to a list[str]."""
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
    """Best-effort JSON extraction. Returns dict with normalized types."""
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
    video_path: str,
    video_id: str,
    vlm,                                    # veil.reasoning.vlm_client.VLMClient
    segment_seconds: float = 16.0,
    fps_per_segment: float = 1.0,
    max_frames_per_segment: int = 8,
    prompt_template: str = DEFAULT_PROMPT,
    progress: bool = True,
) -> MemoryBank:
    """Run VLM over each segment and assemble a MemoryBank."""
    segments: List[VideoSegment] = segment_video(
        video_path,
        segment_seconds=segment_seconds,
        fps_per_segment=fps_per_segment,
        max_frames_per_segment=max_frames_per_segment,
    )
    duration = video_duration(video_path)

    chunks: List[MemoryChunk] = []
    iterator = tqdm(segments, desc=f"build_memory[{video_id}]") if progress else segments
    for seg in iterator:
        prompt = prompt_template.format(n_frames=len(seg.frames), seg_seconds=seg.end_time - seg.start_time, t0=seg.start_time, t1=seg.end_time)
        raw = vlm.chat_with_frames(seg.frames, prompt, max_new_tokens=384)
        parsed = _safe_json(raw)
        chunks.append(
            MemoryChunk(
                video_id=video_id,
                chunk_id=seg.chunk_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                visual_caption=parsed.get("visual_caption", ""),
                event_summary=parsed.get("event_summary", ""),
                objects=parsed.get("objects", []) or [],
                persons=parsed.get("persons", []) or [],
                actions=parsed.get("actions", []) or [],
                ocr=parsed.get("ocr", "") or "",
                memory_text=_make_memory_text(parsed, seg.start_time, seg.end_time),
            )
        )
    return MemoryBank(video_id=video_id, duration=duration, segment_seconds=segment_seconds, chunks=chunks)


def memory_bank_path(cache_dir: str | Path, video_id: str) -> Path:
    return Path(cache_dir) / f"{video_id}.json"
