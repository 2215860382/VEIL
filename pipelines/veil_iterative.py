"""VEIL iterative loop.

Flow:
  Planner(q, opts, missing=None) → {query, rubric}
  loop:
    retrieve(query) → evidence chunks
    Verifier(q, opts, evidence, rubric) → {label, missing_evidence}
    if sufficient  → break
    if no new chunks → break
    Planner(q, opts, missing_evidence) → {query, rubric}  [repair call]
  TextAnswerer(q, opts, evidence) → answer
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from memory.schema import MemoryBank
from reasoning.answerer import TextAnswerer
from reasoning.planner import Planner
from reasoning.verifier import Verifier


def _retrieve_once(
    query: str,
    texts: List[str],
    embedder,
    reranker,
    coarse_k: int,
    rerank_k: int,
) -> List[int]:
    if not texts:
        return []
    q_vec    = embedder.encode([query])[0]
    doc_vecs = embedder.encode(texts)
    scores   = doc_vecs @ q_vec
    k        = min(coarse_k, len(scores))
    coarse   = np.argpartition(-scores, k - 1)[:k]
    coarse   = coarse[np.argsort(-scores[coarse])]
    pairs    = reranker.rerank(query, [texts[i] for i in coarse], top_k=rerank_k)
    return [int(coarse[i]) for i, _ in pairs]


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
    trace = {"queries": [], "rubric": "", "iterations": []}

    if not texts:
        return {"answer": "", "evidence": [], "rationale": "no memory chunks",
                "evidence_texts": [], "evidence_chunk_ids": [], "trace": trace}

    # ── First Planner call: initial query + rubric ─────────────────────────────
    plan   = planner.plan(question, candidates, missing_analysis=None)
    query  = plan["query"]
    rubric = plan["rubric"]
    trace["rubric"] = rubric
    trace["queries"].append(query)

    accumulated_idx: List[int] = []

    for it in range(max_iter):
        retrieved_idx = _retrieve_once(
            query, texts, embedder, reranker, coarse_top_k, rerank_top_k
        )
        new_idx       = [i for i in retrieved_idx if i not in accumulated_idx]
        accumulated_idx = _merge_unique(accumulated_idx, retrieved_idx, cap=final_evidence_cap)
        evidence_texts  = [texts[i] for i in accumulated_idx]

        # ── Verifier: judge sufficiency against rubric ─────────────────────────
        verdict  = verifier.verify(question, candidates, evidence_texts, rubric)
        missing  = verdict.get("missing_evidence")   # None | {type, description}
        trace["iterations"].append({
            "iter":               it,
            "query":              query,
            "retrieved_chunk_ids": [int(bank.chunks[i].chunk_id) for i in retrieved_idx],
            "new_chunk_ids":       [int(bank.chunks[i].chunk_id) for i in new_idx],
            "label":              verdict["label"],
            "missing_evidence":   missing,
        })

        if verdict["label"] == "sufficient":
            break
        if not missing or not new_idx:
            break

        # ── Repair Planner call: target the missing evidence ───────────────────
        repair  = planner.plan(question, candidates, missing_analysis=missing)
        new_q   = repair["query"]
        if not new_q or new_q == query:
            break
        query = new_q
        trace["queries"].append(query)

    # ── Final answer ───────────────────────────────────────────────────────────
    evidence_texts    = [texts[i] for i in accumulated_idx]
    evidence_chunk_ids = [bank.chunks[i].chunk_id for i in accumulated_idx]

    result = llm_answerer.answer(question, candidates, evidence_texts)
    result["evidence_texts"]     = evidence_texts
    result["evidence_chunk_ids"] = evidence_chunk_ids
    result["trace"]              = trace
    return result
