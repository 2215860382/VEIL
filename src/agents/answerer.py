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

_PROMPT_HEADER = (
    "You are answering a multiple-choice question about a long video.\n"
    "Below are {n} relevant video segments retrieved from the video.\n\n"
)

_PROMPT_FOOTER = (
    "\n{hint}"
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


def _build_content(
    prompt: str,
    frames: list,
    image_placement: str,
    evidence_texts: List[str],
    ev_chunk_ids: Sequence,
    kf_chunk_ids: Sequence,
) -> list:
    """Build the OpenAI-style content list for the given image_placement strategy."""
    if image_placement == "text_first":
        content = [{"type": "text", "text": prompt}]
        content += [_img_block(img) for img in frames]
        return content

    if image_placement == "interleaved" and ev_chunk_ids and kf_chunk_ids:
        # Build a map: chunk_id → list of images (in order)
        cid_to_imgs: dict = {}
        for img, cid in zip(frames, kf_chunk_ids):
            cid_to_imgs.setdefault(cid, []).append(img)

        n = len(evidence_texts)
        header = _PROMPT_HEADER.format(n=n)
        footer = _PROMPT_FOOTER  # filled by caller via prompt suffix

        # Extract footer from the full prompt (everything after the last segment)
        # The footer is: hint + Question + Choices + Answer instruction
        # We rebuild it separately from the header + evidence body.
        # For simplicity: split prompt at the evidence block boundary.
        # Since we have evidence_texts, we can reconstruct header and footer directly.
        # (prompt already has the full text; we re-derive the footer from it.)
        # Simpler: just use the pre-built prompt but split at each "--- Segment" boundary.
        # Actually cleanest: re-derive content from scratch without relying on full prompt.

        # Re-derive hint+question+choices from the full prompt by stripping the evidence section.
        # The evidence section is everything between header and "\n\nQuestion:" (or hint).
        # Instead, pass them as already-rendered strings and reconstruct.
        # We already have `prompt` which is the full assembled text.
        # Split: header part | segments | footer part
        # Since prompt = header + evidence_block + "\n\n" + hint + question..., we know:
        #   - header ends after "...from the video.\n\n"
        #   - footer starts at "\n\n" + hint (or "Question:" if no hint)
        # Easiest: find where evidence ends.
        last_seg_marker = f"--- Segment {n} ---\n" if n else ""
        if last_seg_marker and last_seg_marker in prompt:
            after_last = prompt.split(last_seg_marker, 1)[1]
            # after_last = "<last segment text>\n\n<footer>"
            # find the double-newline that separates last segment text from footer
            sep = "\n\n"
            sep_idx = after_last.find(sep)
            if sep_idx >= 0:
                footer_text = after_last[sep_idx + len(sep):]
            else:
                footer_text = after_last
        else:
            # Fallback: no segments, put full prompt as footer
            footer_text = prompt

        content = [{"type": "text", "text": header}]
        for i, (txt, cid) in enumerate(zip(evidence_texts, ev_chunk_ids)):
            seg_text = f"--- Segment {i+1} ---\n{txt}"
            content.append({"type": "text", "text": seg_text})
            for img in cid_to_imgs.get(cid, []):
                content.append(_img_block(img))
        content.append({"type": "text", "text": "\n" + footer_text})
        return content

    # images_first (default / fallback)
    content = [_img_block(img) for img in frames]
    content.append({"type": "text", "text": prompt})
    return content


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
        evidence_chunk_ids=(),
        image_placement: str = "images_first",
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
        prompt   = _PROMPT.format(
            n=len(focused_texts) + len(evidence_texts),
            evidence=evidence,
            hint=hint,
            question=question,
            choices=choices,
        )
        frames = [img for img in keyframe_images if img is not None]

        if image_placement != "images_first" and frames and hasattr(self.model, "chat_with_content"):
            # Combine focused_texts + evidence_texts for interleaved alignment.
            # focused_texts don't have chunk IDs; prepend them without images.
            all_ev_texts = list(focused_texts) + list(evidence_texts)
            # chunk IDs only for the evidence_texts portion; focused_texts get sentinel -1
            all_ev_cids = [-1] * len(focused_texts) + list(evidence_chunk_ids)
            content = _build_content(
                prompt=prompt,
                frames=frames,
                image_placement=image_placement,
                evidence_texts=all_ev_texts,
                ev_chunk_ids=all_ev_cids,
                kf_chunk_ids=keyframe_chunk_ids,
            )
            raw = self.model.chat_with_content(content, max_new_tokens=16, enable_thinking=False)
        else:
            raw = self.model.chat_with_frames(frames, prompt, max_new_tokens=16, enable_thinking=False)

        m = re.search(r"\b([A-D])\b", raw)
        letter = m.group(1) if m else ""
        return {"answer": letter, "evidence": [], "rationale": ""}
