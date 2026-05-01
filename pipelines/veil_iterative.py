"""VEIL: planner → retrieve+rerank → verifier → iterate → Qwen3-8B answerer."""
from __future__ import annotations

from typing import List

import numpy as np

from memory.schema import MemoryBank
from reasoning.answerer import TextAnswerer
from reasoning.planner import Planner
from reasoning.verifier import Verifier


def _retrieve_once(query: str, texts: List[str], embedder, reranker, coarse_k: int, rerank_k: int):
    if not texts:
        return [], []
    q_vec = embedder.encode([query])[0]
    doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec
    k = min(coarse_k, len(scores))
    coarse_idx = np.argpartition(-scores, k - 1)[:k]
    coarse_idx = coarse_idx[np.argsort(-scores[coarse_idx])]

    cand_texts = [texts[i] for i in coarse_idx]
    pairs = reranker.rerank(query, cand_texts, top_k=rerank_k)
    final_idx = [int(coarse_idx[i]) for i, _ in pairs]
    return final_idx, [scores[i] for i in coarse_idx[: rerank_k]]


def _merge_unique(existing: List[int], new: List[int], cap: int) -> List[int]:
    out = list(existing)
    for i in new:
        if i not in out:
            out.append(i)
        if len(out) >= cap:
            break
    return out


def run_veil(
    question: str,
    candidates: List[str],
    bank: MemoryBank,
    planner: Planner,
    verifier: Verifier,
    embedder,
    reranker,
    llm_answerer: TextAnswerer,
    coarse_top_k: int = 50,
    rerank_top_k: int = 10,
    max_iter: int = 3,
    final_evidence_cap: int = 15,
) -> dict:
    texts = bank.memory_texts()
    trace = {"queries": [], "iterations": []}
    if not texts:
        return {"answer": "", "evidence": [], "rationale": "no memory chunks",
                "evidence_texts": [], "evidence_chunk_ids": [], "trace": trace}

    plan = planner.plan(question, candidates)
    query = plan.get("search_query") or question
    trace["queries"].append(query)
    trace["plan"] = plan

    accumulated_idx: List[int] = []
    for it in range(max_iter):
        retrieved_idx, _ = _retrieve_once(query, texts, embedder, reranker, coarse_top_k, rerank_top_k)
        new_idx = [i for i in retrieved_idx if i not in accumulated_idx]
        accumulated_idx = _merge_unique(accumulated_idx, retrieved_idx, cap=final_evidence_cap)
        evidence_texts = [texts[i] for i in accumulated_idx]

        decision = verifier.verify(question, candidates, evidence_texts)
        trace["iterations"].append({
            "iter": it,
            "query": query,
            "retrieved_chunk_ids": [int(bank.chunks[i].chunk_id) for i in retrieved_idx],
            "new_chunk_ids": [int(bank.chunks[i].chunk_id) for i in new_idx],
            "decision": decision,
        })
        if decision.get("is_sufficient"):
            break
        next_q = decision.get("next_query", "").strip()
        if not next_q or next_q == query or not new_idx:
            break
        query = next_q
        trace["queries"].append(query)

    evidence_texts = [texts[i] for i in accumulated_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in accumulated_idx]
    result = llm_answerer.answer(question, candidates, evidence_texts)
    result["evidence_texts"] = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    result["trace"] = trace
    return result
