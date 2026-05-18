"""Coarse retrieval only: BGE-M3 cosine similarity → top-k → answerer (no cross-encoder).

Used for ``coarse8`` / ``coarse64`` (and *_27b text answerer variants).
"""
from __future__ import annotations

from typing import List

import numpy as np

from experiments.pipelines._keyframes import load_keyframes
from memory.schema import MemoryBank


def run_coarse_rag(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    embedder,
    answerer,
    top_k: int = 8,
    siglip=None,
    text_alpha: float = 0.6,
    keyframe_dir=None,
    answer_evidence_cap: int | None = None,
    kf_dedup_threshold: float = 0.92,
    keyframe_cap: int = 8,
    max_evidence_chars: int = 80000,
    rubric_rerank: bool = False,
    rubric: dict | None = None,
    llm=None,
) -> dict:
    texts      = bank.memory_texts()
    texts_ev   = bank.memory_texts(with_time=True, with_asr=True)
    if not texts:
        result = answerer.answer(question, candidates, [])
        result["evidence_texts"] = []
        result["evidence_chunk_ids"] = []
        return result

    opts = "\n".join(f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
    query = f"{question}\nChoices:\n{opts}"
    q_vec = embedder.encode([query])[0]
    if bank.chunks[0].v_semantic:
        doc_vecs = np.array([c.v_semantic for c in bank.chunks], dtype=np.float32)
    else:
        doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec

    if siglip is not None and bank.chunks[0].v_visual:
        vis_query = f"A video frame showing {question} Possible answers: {' or '.join(candidates)}"
        vq_vec    = siglip.encode_text([vis_query])[0]
        vis_vecs  = np.array([c.v_visual for c in bank.chunks], dtype=np.float32)
        vis_scores = vis_vecs @ vq_vec
        scores = text_alpha * scores + (1 - text_alpha) * vis_scores

    k        = min(top_k, len(scores))
    top_idx  = np.argpartition(-scores, k - 1)[:k]
    top_idx  = top_idx[np.argsort(-scores[top_idx])]
    if answer_evidence_cap is not None:
        cap = min(answer_evidence_cap, len(top_idx))
        top_idx = top_idx[:cap]

    if rubric_rerank and llm is not None and rubric is not None and len(top_idx) > 1:
        from experiments.pipelines.veil import _rerank_by_rubric
        ordered_texts = [texts_ev[i] for i in top_idx]
        order = _rerank_by_rubric(question, candidates, ordered_texts, rubric, llm)
        top_idx = top_idx[[int(i) for i in order]]

    evidence_texts     = [texts_ev[i] for i in top_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in top_idx]

    keyframes = []
    if keyframe_dir is not None:
        keyframes = load_keyframes(
            [bank.chunks[i] for i in top_idx], keyframe_dir, bank.video_id,
            dedup_threshold=kf_dedup_threshold, cap=keyframe_cap,
        )

    result = answerer.answer(question, candidates, evidence_texts,
                            keyframe_images=keyframes, max_evidence_chars=max_evidence_chars)
    result["evidence_texts"]     = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    return result
