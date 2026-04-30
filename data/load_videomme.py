"""Loader for Video-MME. Stub — to be wired once data is finalized."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class VideoMMESample:
    video_id: str
    video_path: str
    duration: float
    question: str
    candidates: List[str]
    answer: str
    question_type: str
    subtitle_path: str | None = None


def load_videomme(*args, **kwargs) -> List[VideoMMESample]:
    raise NotImplementedError(
        "Video-MME loader is staged but not implemented yet. "
        "Fill in once /home/Dataset/VideoMME layout is confirmed."
    )
