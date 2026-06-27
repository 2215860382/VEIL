"""Shared records for evidence-sufficiency rubric construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


INITIAL_RUBRIC = [
    {
        "name": "evidence_coverage",
        "description": (
            "Evidence covers the key event, time window, state change, comparison target, "
            "or causal chain needed to verify or exclude each option."
        ),
    },
    {
        "name": "evidence_specificity",
        "description": (
            "Evidence explicitly grounds the relevant person, object, action, scene, speaker, "
            "text, or spatial relation instead of only giving a generic related scene."
        ),
    },
    {
        "name": "evidence_consistency",
        "description": (
            "Evidence is internally consistent and contains enough contrary evidence to exclude "
            "options that conflict with the video."
        ),
    },
]


@dataclass
class DevQuestion:
    sample_idx: int
    video_id: str
    question_id: str
    question_type: str
    question: str
    candidates: list[str]
    priority_reasons: list[str] = field(default_factory=list)
    source_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    chunk_id: int
    start_time: float | None
    end_time: float | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChain:
    sample_idx: int
    video_id: str
    question_type: str
    chain_id: str
    quality: str
    source: str
    evidence: list[EvidenceItem]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


@dataclass
class ChainPair:
    sample_idx: int
    video_id: str
    question_type: str
    pair_id: str
    weaker_chain_id: str
    stronger_chain_id: str
    pair_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

