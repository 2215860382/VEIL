"""Memory chunk schema. One row per fixed-time-window video segment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


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

    def label(self) -> str:
        return f"{self.video_id}#chunk{self.chunk_id:03d}@{self.start_time:.0f}-{self.end_time:.0f}s"


class MemoryBank(BaseModel):
    video_id: str
    duration: float
    segment_seconds: float
    chunks: List[MemoryChunk]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        return cls.model_validate_json(Path(path).read_text())

    def memory_texts(self) -> List[str]:
        return [c.memory_text for c in self.chunks]
