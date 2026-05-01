"""Loader for Video-MME benchmark.

Parquet layout:
  video_id, duration (short/medium/long), domain, sub_category,
  url, videoID, question_id, task_type, question,
  options (["A. text", "B. text", ...]), answer (letter)

Videos live at {video_dir}/{videoID}.mp4
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class VideoMMESample:
    video_id: str           # videoID (YouTube ID)
    video_path: str
    duration_group: str     # short / medium / long
    domain: str
    question_id: str
    question: str
    candidates: List[str]   # stripped of "A. " prefix
    answer: str             # gold answer text (one of candidates)
    sample_idx: int
    duration: float = 0.0


def load_videomme(
    parquet_path: str | Path,
    video_dir: str | Path,
    duration_groups: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> List[VideoMMESample]:
    """Load Video-MME samples from parquet.

    Args:
        parquet_path:    path to test-00000-of-00001.parquet
        video_dir:       dir containing {videoID}.mp4 files
        duration_groups: subset of {"short","medium","long"}; None = all
        max_samples:     cap total samples returned
    """
    import pandas as pd

    video_dir = Path(video_dir)
    df = pd.read_parquet(Path(parquet_path))
    if duration_groups is not None:
        df = df[df["duration"].isin(list(duration_groups))]

    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    samples: List[VideoMMESample] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        raw_opts = list(row.options)
        # Strip leading "A. " / "B. " prefix
        candidates = [o[3:].strip() if len(o) > 2 and o[1] == "." else o for o in raw_opts]
        ans_idx = letter_to_idx.get(str(row.answer).strip().upper(), 0)
        answer_text = candidates[ans_idx] if ans_idx < len(candidates) else str(row.answer)

        samples.append(VideoMMESample(
            video_id=str(row.videoID),
            video_path=str(video_dir / f"{row.videoID}.mp4"),
            duration_group=str(row.duration),
            domain=str(row.domain),
            question_id=str(row.question_id),
            question=str(row.question),
            candidates=candidates,
            answer=answer_text,
            sample_idx=idx,
        ))
    return samples
