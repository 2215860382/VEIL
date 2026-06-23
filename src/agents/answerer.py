"""Answerer — produce final letter-choice given evidence."""
from __future__ import annotations

import re
from typing import List, Sequence


def _format_evidence(evidence_texts: List[str], offset: int = 0) -> str:
    if not evidence_texts:
        return "(no evidence)"
    return "\n".join(f"--- Segment {i+1+offset} ---\n{t}" for i, t in enumerate(evidence_texts))


def _format_options(candidates: List[str]) -> str:
    return " ".join(f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


_PROMPT = (
    "You are answering a multiple-choice question about a long video.\n"
    "Below are {n} relevant video segments retrieved from the video.\n\n"
    "{evidence}\n\n"
    "{hint}"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter (A/B/C/D) only, no explanation:"
)

_QF_HEADER = (
    "You are answering a multiple-choice question about a long video.\n"
    "Question: {question}\n"
    "Choices: {choices}\n\n"
    "Inspect the {n_kf} keyframes and {n_seg} retrieved segments below, "
    "then choose A/B/C/D.\n"
)

_QF_TAIL = (
    "{hint}"
    "{evidence}\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter (A/B/C/D) only, no explanation:"
)


def _format_verifier_hint(option_judgment: dict, option_scores: dict | None = None) -> str:
    """Format verifier's per-option judgement as a prior for the answerer."""
    if not option_judgment:
        return ""
    parts = []
    for letter, verdict in option_judgment.items():
        score_str = ""
        if option_scores and letter in option_scores:
            score_str = f" (score {option_scores[letter]:.2f})"
        parts.append(f"  ({letter}) {verdict}{score_str}")
    return (
        "A separate verifier judged each option using rubric criteria. Use this as a strong prior:\n"
        + "\n".join(parts)
        + "\n\n"
    )


def _img_block(img) -> dict:
    from src.clients.vlm_client import _pil_to_b64
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(img)}"}}


def _image_label(ts: float | None, seg_idx: int | None) -> str:
    """Short text block to prepend before an image when --image-timestamps is on."""
    parts = []
    if ts is not None:
        parts.append(f"at {ts:.0f}s")
    if seg_idx is not None and seg_idx > 0:
        parts.append(f"Segment {seg_idx}")
    return f"[Frame {' — '.join(parts)}]" if parts else "[Frame]"


class Answerer:
    def __init__(self, model):
        self.model = model

    def answer(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        keyframe_images=(),
        keyframe_chunk_ids=(),
        keyframe_ts: Sequence[float] = (),
        evidence_chunk_ids=(),
        max_evidence_chars: int = 80000,
        focused_texts: List[str] = (),
        verifier_option_judgment: dict | None = None,
        verifier_option_scores: dict | None = None,
    ) -> dict:
        all_texts = list(focused_texts) + list(evidence_texts)
        if all_texts and max_evidence_chars:
            per = max_evidence_chars // len(all_texts)
            focused_texts  = [t[:per] for t in focused_texts]
            evidence_texts = [t[:per] for t in evidence_texts]

        evidence = _format_evidence(list(focused_texts) + list(evidence_texts))
        choices  = _format_options(candidates)
        hint     = _format_verifier_hint(verifier_option_judgment or {}, verifier_option_scores)
        n_seg    = len(focused_texts) + len(evidence_texts)

        # Filter to non-null frames, aligning ts and cid lists by position.
        frames = []
        ts_list: List[float | None] = []
        cid_list: List[int] = []
        kf_ts_seq = list(keyframe_ts)
        kf_cid_seq = list(keyframe_chunk_ids)
        for i, img in enumerate(keyframe_images):
            if img is None:
                continue
            frames.append(img)
            ts_list.append(kf_ts_seq[i] if i < len(kf_ts_seq) else None)
            cid_list.append(kf_cid_seq[i] if i < len(kf_cid_seq) else -1)

        # chunk_id → 1-based segment index (for image labels)
        seg_idx_for_cid = {int(cid): i + 1 for i, cid in enumerate(evidence_chunk_ids)}

        content: list = [{
            "type": "text",
            "text": _QF_HEADER.format(
                question=question, choices=choices,
                n_kf=len(frames), n_seg=n_seg,
            ),
        }]
        for img, ts, cid in zip(frames, ts_list, cid_list):
            label = _image_label(ts, seg_idx_for_cid.get(int(cid)))
            content.append({"type": "text", "text": label})
            content.append(_img_block(img))
        content.append({
            "type": "text",
            "text": _QF_TAIL.format(
                hint=hint, evidence=evidence,
                question=question, choices=choices,
            ),
        })
        raw = self.model.chat_with_content(
            content, max_new_tokens=16, enable_thinking=False
        )

        m = re.search(r"\b([A-D])\b", raw)
        letter = m.group(1) if m else ""
        return {"answer": letter, "evidence": [], "rationale": ""}
