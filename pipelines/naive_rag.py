"""Baseline 2: memory bank + BGE retrieve + rerank → VLAnswerer.

Differs from VEIL only in: no planner, no verifier, no iterative loop. Single-shot retrieval.
"""
from __future__ import annotations

from typing import List

from memory.schema import MemoryBank
from reasoning.answerer import VLAnswerer


def run_naive_rag(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    embedder,
    reranker,
    answerer: VLAnswerer,
    coarse_top_k: int = 50,
    rerank_top_k: int = 10,
) -> dict:
    texts = bank.memory_texts()
    if not texts:
        result = answerer.answer(question, candidates, [], evidence_frames=())
        return {**result, "evidence_texts": [], "evidence_chunk_ids": []}

    # Coarse: cosine over BGE-M3 embeddings.
    import numpy as np
    q_vec = embedder.encode([question])[0]
    doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec
    k = min(coarse_top_k, len(scores))
    coarse_idx = np.argpartition(-scores, k - 1)[:k]
    coarse_idx = coarse_idx[np.argsort(-scores[coarse_idx])]

    # Fine: cross-encoder rerank.
    cand_texts = [texts[i] for i in coarse_idx]
    rerank_pairs = reranker.rerank(question, cand_texts, top_k=rerank_top_k)
    final_idx = [int(coarse_idx[i]) for i, _ in rerank_pairs]

    evidence_texts = [texts[i] for i in final_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in final_idx]

    result = answerer.answer(question, candidates, evidence_texts, evidence_frames=())
    result["evidence_texts"] = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    return result
