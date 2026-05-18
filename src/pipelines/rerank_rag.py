"""Coarse BGE retrieval + cross-encoder rerank → answerer (single-shot, no VEIL loop)."""
from __future__ import annotations

from typing import List

import numpy as np

from src.pipelines._keyframes import load_keyframes
from src.memory.core.schema import MemoryBank


def run_rerank_rag(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    embedder,
    reranker,
    answerer,
    coarse_top_k: int = 50,
    rerank_top_k: int = 10,
    siglip=None,
    text_alpha: float = 0.6,
    keyframe_dir=None,
    kf_dedup_threshold: float = 0.92,
    keyframe_cap: int = 8,
) -> dict:
    texts    = bank.memory_texts()
    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
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
        vis_query  = f"A video frame showing {question} Possible answers: {' or '.join(candidates)}"
        vq_vec     = siglip.encode_text([vis_query])[0]
        vis_vecs   = np.array([c.v_visual for c in bank.chunks], dtype=np.float32)
        vis_scores = vis_vecs @ vq_vec
        scores = text_alpha * scores + (1 - text_alpha) * vis_scores

    k = min(coarse_top_k, len(scores))
    coarse_idx = np.argpartition(-scores, k - 1)[:k]
    coarse_idx = coarse_idx[np.argsort(-scores[coarse_idx])]

    rerank_pairs = reranker.rerank(query, [texts_ev[i] for i in coarse_idx], top_k=rerank_top_k)
    final_idx = [int(coarse_idx[i]) for i, _ in rerank_pairs]

    evidence_texts     = [texts_ev[i] for i in final_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in final_idx]

    keyframes = []
    if keyframe_dir is not None:
        keyframes = load_keyframes(
            [bank.chunks[i] for i in final_idx], keyframe_dir, bank.video_id,
            dedup_threshold=kf_dedup_threshold, cap=keyframe_cap,
        )

    result = answerer.answer(question, candidates, evidence_texts, keyframe_images=keyframes)
    result["evidence_texts"]     = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    return result
