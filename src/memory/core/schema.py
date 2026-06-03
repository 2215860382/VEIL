"""Memory chunk schema. One row per sliding-window chunk of pre-sampled frames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class StaticAttributeFrame(BaseModel):
    """Static visual attributes extracted from a single frame."""
    frame_id: str
    timestamp: float
    image_path: str
    ocr_text: List[str] = Field(default_factory=list)
    numbers: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    object_attributes: List[dict] = Field(default_factory=list)
    people_appearance: List[str] = Field(default_factory=list)
    clothing: List[str] = Field(default_factory=list)
    spatial_layout: List[str] = Field(default_factory=list)
    textures: List[str] = Field(default_factory=list)
    scene_attributes: List[str] = Field(default_factory=list)


class MemoryChunk(BaseModel):
    video_id: str
    chunk_id: int
    start_time: float
    end_time: float
    memory_text: str                       # the canonical retrieval-target text
    visual_caption: str = ""               # raw VL caption of frames
    event_summary: str = ""
    objects: List[str] = Field(default_factory=list)
    persons: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    ocr: str = ""
    asr: str = ""
    # Event-level / dense baselines (optional; empty for legacy frame-window banks).
    memory_schema: str = ""
    memory_id: str = ""
    time_range: str = ""
    sampled_frames: List[float] = Field(default_factory=list)
    scene: str = ""
    state_change: str = ""
    temporal_relation: str = ""
    evidence_caption: str = ""
    # Chunk fields for similarity_group banks (SigLIP cosine grouping + BGE/VLM text)
    v_semantic: List[float] = Field(default_factory=list)   # BGE-M3 (1024,) L2-normed
    v_visual: List[float] = Field(default_factory=list)     # SigLIP keyframe (1024,) L2-normed
    keyframe_path: str = ""
    keyframe_ts: float = 0.0

    # Two-layer memory bank fields (Dynamic Narrative + Static Attribute)
    key_events: List[str] = Field(default_factory=list)
    actors: List[str] = Field(default_factory=list)
    state_changes: List[str] = Field(default_factory=list)
    temporal_relations: List[str] = Field(default_factory=list)
    causal_clues: List[str] = Field(default_factory=list)
    static_frames: List[StaticAttributeFrame] = Field(default_factory=list)
    static_index_text: str = ""
    v_static: List[float] = Field(default_factory=list)   # BGE-M3 on static attributes

    def label(self) -> str:
        return f"{self.video_id}#chunk{self.chunk_id:03d}@{self.start_time:.0f}-{self.end_time:.0f}s"


class MemoryBank(BaseModel):
    video_id: str
    duration: float
    chunks: List[MemoryChunk]
    # Sampling provenance (set on build; None for legacy banks).
    chunk_size: Optional[int] = None
    stride: Optional[int] = None
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    resolution: Optional[int] = None
    # Legacy field from old time-based segmenter; kept for loading old banks.
    segment_seconds: Optional[float] = None
    # event_v1 | dense_1fps_v1 | frame_window_v1 (legacy) | similarity_group
    memory_kind: str = "frame_window_v1"
    # Effective temporal chunk length (seconds) for event / dense builders.
    chunk_sec: Optional[float] = None
    stride_sec: Optional[float] = None
    # Provenance from src.memory.similarity (similarity-based grouping).
    vlm_caption_model: Optional[str] = None
    vlm_summary_model: Optional[str] = None
    vlm_caption_backend: Optional[str] = None  # "local" | "api"
    vlm_summary_backend: Optional[str] = None
    vlm_api_base_url: Optional[str] = None
    siglip_model: Optional[str] = None
    bge_model: Optional[str] = None
    # Per-video subtitle provenance (VideoMME / any benchmark with --subtitle-dir).
    subtitle_dir: Optional[str] = None
    subtitle_file_present: Optional[bool] = None
    subtitle_cue_count: Optional[int] = None
    # Fixed-frame bank metadata (src.memory.fixedframe).
    frames_per_chunk: Optional[int] = None
    frame_positions_within_chunk: Optional[List[float]] = None
    chunking_readme_zh: Optional[str] = None
    chunking_readme_en: Optional[str] = None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        return cls.model_validate_json(Path(path).read_text())

    def memory_texts(self, with_time: bool = False, with_asr: bool = False, with_layers: bool = False) -> List[str]:
        results = []
        for c in self.chunks:
            if with_layers:
                parts = []
                if with_time:
                    parts.append(f"[{c.start_time:.0f}s-{c.end_time:.0f}s]")
                parts.append(c.memory_text)
                if c.key_events:
                    parts.append(f"Events: {' | '.join(c.key_events)}")
                if c.actors:
                    parts.append(f"Actors: {' | '.join(c.actors)}")
                if c.state_changes:
                    parts.append(f"Changes: {' | '.join(c.state_changes)}")
                if c.static_index_text:
                    parts.append(f"Visual: {c.static_index_text}")
                if with_asr and c.asr.strip():
                    parts.append(f"Speech: {c.asr}")
                text = "\n".join(parts)
            else:
                text = f"[{c.start_time:.0f}s-{c.end_time:.0f}s] {c.memory_text}" if with_time else c.memory_text
                if with_asr and c.asr.strip():
                    text = f"{text}\nSpeech: {c.asr}"
            results.append(text)
        return results
