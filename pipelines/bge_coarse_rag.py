"""Baseline: BGE-M3 coarse retrieval only (no reranker) → Qwen3-8B.

Single-stage dense retrieval. Differs from bge_rerank in that no cross-encoder
reranking is applied — only cosine similarity over BGE-M3 embeddings.
"""
from __future__ import annotations

from typing import List

import numpy as np

from memory.schema import MemoryBank
from reasoning.answerer import TextAnswerer


def run_bge_coarse_rag(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    embedder,
    llm_answerer: TextAnswerer,
    top_k: int = 10,
) -> dict:
    """Answer by retrieving top-k chunks via BGE-M3 cosine similarity.

    Args:
        question:      Question text.
        candidates:    Answer option strings.
        bank:          Memory bank.
        embedder:      BGEM3Embedder (encode returns unit-normed vectors).
        llm_answerer:  TextAnswerer backed by Qwen3-8B.
        top_k:         Number of chunks to retrieve.
    """
    texts = bank.memory_texts()
    if not texts:
        result = llm_answerer.answer(question, candidates, [])
        result["evidence_texts"] = []
        result["evidence_chunk_ids"] = []
        return result

    q_vec = embedder.encode([question])[0]
    doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec

    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])].tolist()

    evidence_texts = [texts[i] for i in top_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in top_idx]

    result = llm_answerer.answer(question, candidates, evidence_texts)
    result["evidence_texts"] = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    return result
