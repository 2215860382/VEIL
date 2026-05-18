"""VEIL — iterative evidence retrieval main loop.

Planner (iter-0 decomposition + iter≥1 unified planner + query safety nets) lives in
``reasoning/planner.py``. Verifier lives in ``reasoning/verifier.py``. This file owns
only the main loop, retrieval primitives, evidence dedup, oracle override, and the
rubric-based final evidence rerank.

Per-iteration retrieval: each sub-question independently fetches BGE coarse top-k
(optionally SigLIP-fused) → optional reranker top-k → de-dup against accumulated
evidence (BGE cosine ≥ ``ev_dedup_threshold``).

Call-site variants (``run_experiments.py --pipelines``):
  ``veil_coarse8``  : coarse top-8 only (no reranker)
  ``veil_coarse64`` : coarse top-64 only (no reranker)
  ``veil_rerank8``  : coarse pool → reranker top-8
"""
from __future__ import annotations

import re
from typing import List, Optional, Set

import numpy as np

from src.pipelines._keyframes import keyframe_path, load_keyframe_pil
from src.memory.core.schema import MemoryBank
from src.reasoning.planner import Planner
from src.reasoning.verifier import Verifier, get_rubric_dict


# ── Broadcast (uniform timeline sampling) ────────────────────────────────────

def _broadcast_retrieve(
    bank: MemoryBank,
    seen_ids: Set[int],
    n: int = 8,
) -> tuple[list, list, list]:
    """Uniformly sample n unseen chunks across the video timeline."""
    texts_ev  = bank.memory_texts(with_time=True, with_asr=True)
    idx_map   = {c.chunk_id: i for i, c in enumerate(bank.chunks)}
    unseen    = [c for c in bank.chunks if c.chunk_id not in seen_ids]
    if not unseen:
        return [], [], []
    unseen.sort(key=lambda c: c.chunk_id)
    step   = max(1, len(unseen) // n)
    chosen = unseen[::step][:n]
    texts, ids, vecs = [], [], []
    for c in chosen:
        i = idx_map.get(c.chunk_id)
        if i is not None:
            texts.append(texts_ev[i])
            ids.append(c.chunk_id)
            vecs.append(c.v_semantic or [])
    return texts, ids, vecs


# ── Oracle mode (debug upper bound) ──────────────────────────────────────────

_ORACLE_MISSING_SYS = """\
You are a retrieval query analyst for a video question-answering system.
The answerer was given evidence but predicted the WRONG answer.
Your job: write ONE concrete, actionable sentence describing what specific information is MISSING from the evidence that would support the correct answer.
Be specific: name the exact fact, time range, event, or comparison that is absent.
Output ONLY a JSON object:
{
  "description": "<one sentence: the specific actionable fact that is absent>"
}"""


def _oracle_analyze_missing(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    gold_letter: str,
    pred_letter: str,
    llm,
) -> dict:
    """Given oracle knowledge of correct answer, ask LLM what evidence is missing."""
    opts  = "\n".join(f"  ({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev    = "\n".join(f"  [E{i+1}] {t[:250]}" for i, t in enumerate(evidence_texts))
    gold_text = candidates[ord(gold_letter) - 65] if gold_letter else "?"
    pred_text = candidates[ord(pred_letter) - 65] if pred_letter and pred_letter != "?" else "unknown"
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        f"Evidence:\n{ev}\n\n"
        f"Correct answer: ({gold_letter}) {gold_text}\n"
        f"Model predicted: ({pred_letter}) {pred_text}\n\n"
        "What specific information is MISSING from the evidence to support the correct answer?"
    )
    messages = [
        {"role": "system", "content": _ORACLE_MISSING_SYS},
        {"role": "user",   "content": user},
    ]
    import json as _json
    raw = llm.chat(messages, max_new_tokens=120, enable_thinking=False)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            d = _json.loads(m.group())
            desc = str(d.get("description", ""))
            return desc if desc else raw[:200]
        except Exception:
            pass
    return raw[:200]


# ── Rubric-based final reranking ─────────────────────────────────────────────

def _rerank_by_rubric(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    rubric: dict,
    llm,
) -> List[int]:
    """Ask LLM to rank evidence chunks by rubric relevance. Returns sorted indices (best first)."""
    if len(evidence_texts) <= 1:
        return list(range(len(evidence_texts)))

    from src.reasoning.verifier import _format_rubric_as_text
    opts    = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
    ev_fmt  = "\n".join(f"[E{i+1}] {t[:300]}" for i, t in enumerate(evidence_texts))
    rubric_text = _format_rubric_as_text(rubric)

    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        f"Evidence chunks:\n{ev_fmt}\n\n"
        "Rank the evidence chunks from MOST to LEAST useful for satisfying the rubric and answering the question.\n"
        "Output ONLY a JSON array of 0-based indices, e.g.: [2, 0, 3, 1]\n"
        "Every index must appear exactly once."
    )
    messages = [{"role": "user", "content": user}]
    raw = llm.chat(messages, max_new_tokens=120, enable_thinking=False)

    import json as _json
    m = re.search(r'\[[\d,\s]+\]', raw)
    if m:
        try:
            order = _json.loads(m.group())
            n = len(evidence_texts)
            valid = [i for i in order if isinstance(i, int) and 0 <= i < n]
            seen  = set(valid)
            missing = [i for i in range(n) if i not in seen]
            return valid + missing
        except Exception:
            pass
    return list(range(len(evidence_texts)))


# ── Chunk-level evidence dedup ───────────────────────────────────────────────

def _dedup_by_similarity(
    new_texts: List[str],
    new_ids:   List[int],
    new_vecs:  List[List[float]],
    existing_vecs: List[List[float]],
    threshold: float = 0.85,
) -> tuple[List[str], List[int], List[List[float]]]:
    """Drop new chunks whose v_semantic cosine sim ≥ threshold against any already-accumulated chunk."""
    if not new_vecs or all(not v for v in new_vecs):
        return new_texts, new_ids, new_vecs
    kept_t, kept_i, kept_v = [], [], []
    running = list(existing_vecs)
    for text, cid, vec in zip(new_texts, new_ids, new_vecs):
        if not vec:
            kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
            continue
        v = np.array(vec, dtype=np.float32)
        if running:
            mat = np.array(running, dtype=np.float32)
            if float(np.max(mat @ v)) >= threshold:
                continue
        kept_t.append(text); kept_i.append(cid); kept_v.append(vec)
        running.append(vec)
    return kept_t, kept_i, kept_v


# ── Query-driven retrieval (per sub-question) ────────────────────────────────

def _query_retrieve(
    query: str,
    bank: MemoryBank,
    embedder,
    reranker,
    coarse_top_k: int,
    final_top_k: int,
    exclude_ids: Set[int],
    siglip=None,
    text_alpha: float = 0.6,
    vis_query: str = "",
) -> tuple[List[str], List[int], List[List[float]]]:
    """Return (evidence_texts, chunk_ids, v_semantic_vecs), skipping already-seen chunks."""
    texts    = bank.memory_texts()
    texts_ev = bank.memory_texts(with_time=True, with_asr=True)
    q_vec  = embedder.encode([query])[0]
    if bank.chunks[0].v_semantic:
        doc_vecs = np.array([c.v_semantic for c in bank.chunks], dtype=np.float32)
    else:
        doc_vecs = embedder.encode(texts)
    scores = doc_vecs @ q_vec

    if siglip is not None and vis_query and bank.chunks[0].v_visual:
        vq_vec     = siglip.encode_text([vis_query])[0]
        vis_vecs   = np.array([c.v_visual for c in bank.chunks], dtype=np.float32)
        vis_scores = vis_vecs @ vq_vec
        scores = text_alpha * scores + (1 - text_alpha) * vis_scores

    k      = min(coarse_top_k, len(scores))
    coarse = np.argpartition(-scores, k - 1)[:k]
    coarse = coarse[np.argsort(-scores[coarse])]

    if reranker is not None:
        reranked  = reranker.rerank(query, [texts_ev[i] for i in coarse], top_k=final_top_k)
        final_idx = [int(coarse[i]) for i, _ in reranked]
    else:
        final_idx = list(coarse[:final_top_k])

    new_idx = [i for i in final_idx if bank.chunks[i].chunk_id not in exclude_ids]
    return (
        [texts_ev[i] for i in new_idx],
        [bank.chunks[i].chunk_id for i in new_idx],
        [bank.chunks[i].v_semantic for i in new_idx],
    )


# ── Main loop ────────────────────────────────────────────────────────────────

def run_veil(
    question:      str,
    candidates:    List[str],
    bank:          MemoryBank,
    embedder,
    answerer,
    llm,
    task_type:     Optional[str] = None,
    reranker                     = None,
    coarse_top_k:  int           = 64,
    final_top_k:   int           = 8,
    max_iter:      int           = 3,
    dedup_thresh:  float         = 0.5,
    siglip                       = None,
    text_alpha:    float         = 0.6,
    keyframe_dir                 = None,
    kf_dedup_threshold:    float = 0.92,
    ev_dedup_threshold:    float = 0.85,
    query_drift_threshold: float | None = 0.70,
    answer_evidence_cap:   int | None   = None,
    keyword_broadcast:     bool         = True,
    use_option_status:     bool         = True,
    prune_satisfied:       bool         = False,
    use_rubric_judgment:   bool         = True,
    chunks_per_iter:       Optional[int] = None,
    per_subq_k:            Optional[int] = None,
    use_subquestions:      bool         = False,
    force_option_subquestions: bool     = False,
    rubric_rerank:         bool         = False,
    evidence_attribution:  bool         = False,
    use_oracle:            bool         = False,
    gold_answer:           str          = "",
) -> dict:
    """Iter-0 sub-question decomposition + iter ≥1 unified-planner repair.

    Each sub-question independently retrieves ``final_top_k`` chunks (no budget splitting);
    iter ≥1 query generation goes through Jaccard (``dedup_thresh``) + BGE drift
    (``query_drift_threshold``) filters; if all targeted queries are filtered, the iter
    falls back to broadcast.
    """
    verifier = Verifier(llm)
    planner  = Planner(llm)
    rubric   = get_rubric_dict(question, task_type)

    vis_query = f"A video frame showing {question} Possible answers: {' or '.join(candidates)}"

    all_evidence_texts: List[str]         = []
    all_chunk_ids:      List[int]         = []
    all_ev_vecs:        List[List[float]] = []
    all_keyframes:      List              = []
    all_kf_vecs:        List[List[float]] = []
    seen_ids:           Set[int]          = set()
    prev_queries:       List[str]         = [question]
    plan_history:       List[dict]        = []
    iterations:         List[dict]        = []
    chunk_by_id = {c.chunk_id: c for c in bank.chunks}

    # ── Iter-0 plan: sub-question decomposition ──────────────────────────────
    current_plan: dict = planner.decompose_iter0(
        question, candidates, force_option=force_option_subquestions,
    )

    for it in range(max_iter):
        do_broadcast      = (current_plan["strategy"] == "broadcast")
        queries_this_iter = current_plan["queries"] if not do_broadcast else []

        # ── Retrieve ──────────────────────────────────────────────────────────
        new_texts, new_ids, new_vecs = [], [], []

        if do_broadcast:
            new_texts, new_ids, new_vecs = _broadcast_retrieve(bank, seen_ids, n=final_top_k)
            new_texts, new_ids, new_vecs = _dedup_by_similarity(
                new_texts, new_ids, new_vecs, all_ev_vecs, ev_dedup_threshold)
            seen_ids.update(new_ids)
            iter_query = "[broadcast]"
        else:
            # Each sub-question independently retrieves final_top_k chunks.
            for q in queries_this_iter:
                vq = vis_query if it == 0 else q
                t, i, v = _query_retrieve(
                    q, bank, embedder, reranker,
                    coarse_top_k, final_top_k, seen_ids,
                    siglip=siglip, text_alpha=text_alpha, vis_query=vq,
                )
                t, i, v = _dedup_by_similarity(t, i, v, all_ev_vecs + new_vecs, ev_dedup_threshold)
                new_texts.extend(t); new_ids.extend(i); new_vecs.extend(v)
                seen_ids.update(i)
            iter_query = (" | ".join(queries_this_iter) if len(queries_this_iter) > 1
                          else (queries_this_iter[0] if queries_this_iter else ""))

        new_texts, new_ids, new_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, all_ev_vecs, ev_dedup_threshold,
        )
        all_evidence_texts.extend(new_texts)
        all_chunk_ids.extend(new_ids)
        all_ev_vecs.extend(new_vecs)

        # ── Keyframes (visual de-dup against all accumulated) ────────────────
        if keyframe_dir is not None:
            for cid in new_ids:
                c = chunk_by_id.get(cid)
                if c is None:
                    continue
                img = load_keyframe_pil(keyframe_path(keyframe_dir, bank.video_id, c.chunk_id))
                if img is None:
                    continue
                if c.v_visual and all_kf_vecs:
                    v = np.array(c.v_visual, dtype=np.float32)
                    if float(np.max(np.array(all_kf_vecs, dtype=np.float32) @ v)) >= kf_dedup_threshold:
                        continue
                all_keyframes.append(img)
                all_kf_vecs.append(c.v_visual if c.v_visual else [])

        # ── Verifier ─────────────────────────────────────────────────────────
        verdict = verifier.verify(question, candidates, all_evidence_texts, rubric,
                                  keyframe_images=all_keyframes,
                                  use_rubric_judgment=use_rubric_judgment,
                                  include_evidence_attribution=evidence_attribution)

        # ── Oracle mode: override label/missing based on gold answerer check ─
        if use_oracle and gold_answer:
            gold_letter = chr(65 + candidates.index(gold_answer)) if gold_answer in candidates else ""
            if len(all_evidence_texts) > 1:
                _order = _rerank_by_rubric(question, candidates, all_evidence_texts, rubric, llm)
                _ev_sorted = [all_evidence_texts[i] for i in _order]
            else:
                _ev_sorted = all_evidence_texts
            _oracle_result = answerer.answer(question, candidates, _ev_sorted,
                                             keyframe_images=all_keyframes)
            _pred = _oracle_result.get("answer", "")
            if _pred == gold_letter:
                verdict["label"] = "sufficient"
                verdict["missing_evidence_analysis"] = None
                verdict["reasoning"] = f"Oracle: pred={_pred} == gold ({verdict.get('reasoning','')})"
            else:
                _missing = _oracle_analyze_missing(
                    question, candidates, all_evidence_texts, gold_letter, _pred, llm)
                verdict["label"] = "insufficient"
                verdict["missing_evidence_analysis"] = _missing
                verdict["reasoning"] = f"Oracle: pred={_pred} != gold={gold_letter} ({verdict.get('reasoning','')})"

        # ── Evidence attribution: remove distractors from accumulated set ────
        if evidence_attribution:
            dist_set = {i - 1 for i in (verdict.get("distractor_ids") or [])
                        if 1 <= i <= len(all_evidence_texts)}
            if dist_set:
                keep = [i for i in range(len(all_evidence_texts)) if i not in dist_set]
                all_evidence_texts = [all_evidence_texts[i] for i in keep]
                all_chunk_ids      = [all_chunk_ids[i]      for i in keep]
                all_ev_vecs        = [all_ev_vecs[i]        for i in keep]
                if all_keyframes:
                    all_keyframes  = [all_keyframes[i]      for i in keep]

        # ── Trace ────────────────────────────────────────────────────────────
        iterations.append({
            "iter":                       it,
            "query":                      iter_query,
            "new_ids":                    new_ids,
            "verdict":                    verdict["label"],
            "score":                      verdict.get("score"),
            "criteria":                   verdict.get("criteria"),
            "reasoning":                  verdict.get("reasoning"),
            "missing_evidence_analysis":  verdict.get("missing_evidence_analysis") or None,
            "key_ids":                    verdict.get("key_ids"),
            "distractor_ids":             verdict.get("distractor_ids"),
            "option_status":              verdict.get("option_status"),
            "plan_strategy":              current_plan.get("strategy"),
            "plan_reasoning":             current_plan.get("reasoning"),
        })

        if verdict["label"] == "sufficient":
            break
        if it == max_iter - 1:
            break

        # ── Plan next iter ───────────────────────────────────────────────────
        next_plan, m_desc, _failed = planner.plan_next(
            question, candidates, all_evidence_texts, verdict, plan_history,
            keyword_broadcast=keyword_broadcast,
            use_rubric_judgment=use_rubric_judgment,
            use_option_status=use_option_status,
            prune_satisfied=prune_satisfied,
        )

        # ── Filter targeted queries; fall back to broadcast if all dropped ───
        next_plan = planner.filter_targeted_queries(
            next_plan, prev_queries, embedder, all_ev_vecs,
            dedup_thresh=dedup_thresh,
            drift_threshold=query_drift_threshold,
        )

        plan_history.append({
            "strategy":         next_plan["strategy"],
            "queries":          next_plan["queries"],
            "missing_analysis": m_desc,
        })
        prev_queries.extend(next_plan["queries"])
        current_plan = next_plan

    # ── Answer ───────────────────────────────────────────────────────────────
    ev_texts = all_evidence_texts
    ev_ids   = all_chunk_ids
    if answer_evidence_cap is not None:
        cap      = min(answer_evidence_cap, len(ev_texts))
        ev_texts = ev_texts[:cap]
        ev_ids   = ev_ids[:cap]

    if rubric_rerank and len(ev_texts) > 1:
        order    = _rerank_by_rubric(question, candidates, ev_texts, rubric, llm)
        ev_texts = [ev_texts[i] for i in order]
        ev_ids   = [ev_ids[i]   for i in order]

    result = answerer.answer(question, candidates, ev_texts,
                             keyframe_images=all_keyframes)
    result["evidence_chunk_ids"] = ev_ids
    result["evidence_texts"]     = ev_texts
    result["trace_iters"]        = iterations
    return result
