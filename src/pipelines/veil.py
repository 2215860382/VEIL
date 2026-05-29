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
    coarse_top_k:  int           = 8,
    final_top_k:   int           = 8,
    max_iter:      int           = 3,
    dedup_thresh:  float         = 0.9,
    siglip                       = None,
    text_alpha:    float         = 0.6,
    keyframe_dir                 = None,
    kf_dedup_threshold:    float = 0.92,
    ev_dedup_threshold:    float = 0.85,
    query_drift_threshold: float | None = 0.70,
    answer_evidence_cap:   int | None   = None,
    answer_keyframe_cap:   int | None   = 16,
    verifier_evidence_cap: int | None   = None,
    prune_satisfied:       bool         = False,
    rubric_judgment:       bool         = True,
    force_option_subquestions: bool     = False,
    verifier_attr:         bool         = False,
    verifier_opstatus:     bool         = False,
    rubric_rerank:         bool         = False,
    explicit_attribution:  bool         = False,
    prune_distractors:     bool         = False,
    use_oracle:            bool         = False,
    gold_answer:           str          = "",
    oracle_no_second_rerank: bool       = False,
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
    all_kf_chunk_ids:   List[int]         = []
    seen_ids:           Set[int]          = set()
    plan_history:       List[List[str]]   = [[question]]  # seed with original question for dedup
    iterations:         List[dict]        = []
    chunk_by_id = {c.chunk_id: c for c in bank.chunks}

    # Pruning distractors requires distractor_ids from the verifier.
    # verifier_opstatus already outputs distractor_ids; only fall back to explicit_attribution otherwise.
    if prune_distractors and not explicit_attribution and not verifier_opstatus:
        explicit_attribution = True

    # ── Iter-0 plan: sub-question decomposition ──────────────────────────────
    current_plan: dict = planner.decompose_iter0(
        question, candidates,
        force_option=force_option_subquestions,
    )
    plan_history.append(current_plan["queries"])
    last_verdict: dict = {}
    gold_letter = (chr(65 + candidates.index(gold_answer))
                   if use_oracle and gold_answer and gold_answer in candidates else "")
    oracle_frozen_result: Optional[dict] = None

    for it in range(max_iter):

        # ── Plan (iter ≥ 1): generate new queries from last verdict + history ─
        if it > 0:
            next_plan, m_desc, _weak = planner.plan_next(
                question, candidates, all_evidence_texts, last_verdict, plan_history,
                rubric_judgment=rubric_judgment,
                prune_satisfied=prune_satisfied,
            )
            next_plan = planner.filter_targeted_queries(
                next_plan, plan_history, embedder, all_ev_vecs,
                dedup_thresh=dedup_thresh,
                drift_threshold=query_drift_threshold,
            )
            plan_history.append(next_plan["queries"])
            current_plan = next_plan

        do_broadcast      = (current_plan["strategy"] == "broadcast")
        queries_this_iter = current_plan["queries"] if not do_broadcast else []

        # ── Retrieve ──────────────────────────────────────────────────────────
        new_texts, new_ids, new_vecs = [], [], []

        if do_broadcast:
            new_texts, new_ids, new_vecs = _broadcast_retrieve(bank, seen_ids, n=final_top_k)
            iter_query = "[broadcast]"
        else:
            for q in queries_this_iter:
                vq = vis_query if it == 0 else q
                t, i, v = _query_retrieve(
                    q, bank, embedder, reranker,
                    coarse_top_k, final_top_k, seen_ids,
                    siglip=siglip, text_alpha=text_alpha, vis_query=vq,
                )
                new_texts.extend(t); new_ids.extend(i); new_vecs.extend(v)
                seen_ids.update(i)
            iter_query = (" | ".join(queries_this_iter) if len(queries_this_iter) > 1
                          else (queries_this_iter[0] if queries_this_iter else ""))

        new_texts, new_ids, new_vecs = _dedup_by_similarity(
            new_texts, new_ids, new_vecs, all_ev_vecs, ev_dedup_threshold)
        seen_ids.update(new_ids)
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
                all_kf_chunk_ids.append(cid)

        # ── Verifier ──────────────────────────────────────────────────────────
        if verifier_evidence_cap is not None and len(all_evidence_texts) > verifier_evidence_cap:
            order = _rerank_by_rubric(question, candidates, all_evidence_texts, rubric, llm)
            top_k = min(verifier_evidence_cap, len(all_evidence_texts))
            ev_for_verifier = [all_evidence_texts[i] for i in order[:top_k]]
        else:
            ev_for_verifier = all_evidence_texts

        kf_for_verifier = all_keyframes
        if answer_keyframe_cap is not None and len(kf_for_verifier) > answer_keyframe_cap:
            kf_for_verifier = kf_for_verifier[:answer_keyframe_cap]

        last_verdict = verifier.verify(question, candidates, ev_for_verifier, rubric,
                                  keyframe_images=kf_for_verifier,
                                  rubric_judgment=rubric_judgment,
                                  explicit_attribution=explicit_attribution,
                                  verifier_attr=verifier_attr,
                                  verifier_opstatus=verifier_opstatus)

        # ── Oracle check (post-verifier) ──────────────────────────────────────
        # Oracle uses the same evidence/keyframe caps as the final answerer.
        # The stable variant can freeze this exact answer order to avoid a second
        # rerank changing the final prediction after oracle already succeeded.
        oracle_pred = ""
        if use_oracle and gold_letter:
            oracle_ev_texts = list(all_evidence_texts)
            oracle_ev_ids   = list(all_chunk_ids)
            if len(oracle_ev_texts) > 1:
                _order = _rerank_by_rubric(question, candidates, all_evidence_texts, rubric, llm)
                oracle_ev_texts = [oracle_ev_texts[i] for i in _order]
                oracle_ev_ids   = [oracle_ev_ids[i]   for i in _order]
            if answer_evidence_cap is not None:
                top_k = min(answer_evidence_cap, len(oracle_ev_texts))
                oracle_ev_texts = oracle_ev_texts[:top_k]
                oracle_ev_ids   = oracle_ev_ids[:top_k]

            kf_for_oracle = all_keyframes
            if answer_keyframe_cap is not None and len(kf_for_oracle) > answer_keyframe_cap:
                kf_for_oracle = kf_for_oracle[:answer_keyframe_cap]

            oracle_pred = answerer.answer(question, candidates, oracle_ev_texts,
                                          keyframe_images=kf_for_oracle).get("answer", "")

            last_verdict["reasoning"] = (
                f"Oracle: pred={oracle_pred or '?'} "
                f"{'==' if oracle_pred == gold_letter else '!='} gold={gold_letter}"
            )
            if oracle_pred == gold_letter:
                last_verdict["label"] = "FULLY_SUFFICIENT"
                last_verdict["missing_evidence_analysis"] = None
                last_verdict["unknown_options"] = []
                if oracle_no_second_rerank:
                    oracle_frozen_result = {
                        "answer": oracle_pred,
                        "evidence_texts": oracle_ev_texts,
                        "evidence_chunk_ids": oracle_ev_ids,
                    }
            else:
                last_verdict["label"] = "INSUFFICIENT"
                last_verdict["missing_evidence_analysis"] = _oracle_analyze_missing(
                    question, candidates, all_evidence_texts, gold_letter, oracle_pred, llm)
                last_verdict["option_judgment"] = {
                    chr(65 + i): ("unknown" if chr(65 + i) == gold_letter else "false")
                    for i in range(len(candidates))
                }
                last_verdict["unknown_options"] = [gold_letter]

        # ── Prune distractors flagged by the verifier ────────────────────────
        if prune_distractors:
            dist_set = {i - 1 for i in (last_verdict.get("distractor_ids") or [])
                        if 1 <= i <= len(all_evidence_texts)}
            if dist_set:
                keep = [i for i in range(len(all_evidence_texts)) if i not in dist_set]
                all_evidence_texts = [all_evidence_texts[i] for i in keep]
                all_chunk_ids      = [all_chunk_ids[i]      for i in keep]
                all_ev_vecs        = [all_ev_vecs[i]        for i in keep]
                if all_keyframes:
                    keep_cids = set(all_chunk_ids)
                    kf_keep = [j for j, cid in enumerate(all_kf_chunk_ids) if cid in keep_cids]
                    all_keyframes    = [all_keyframes[j]    for j in kf_keep]
                    all_kf_vecs      = [all_kf_vecs[j]      for j in kf_keep]
                    all_kf_chunk_ids = [all_kf_chunk_ids[j] for j in kf_keep]

        # ── Trace ────────────────────────────────────────────────────────────
        iterations.append({
            "iter":                      it,
            "query":                     iter_query,
            "new_ids":                   new_ids,
            "verdict":                   last_verdict["label"],
            "option_judgment":           last_verdict.get("option_judgment"),
            "unknown_options":           last_verdict.get("unknown_options"),
            "option_rubric_scores":      last_verdict.get("option_rubric_scores"),
            "weak_rubric_criteria":      last_verdict.get("weak_rubric_criteria"),
            "missing_evidence_analysis": last_verdict.get("missing_evidence_analysis") or None,
            "oracle_pred":               oracle_pred or None,
            "plan_strategy":             current_plan.get("strategy"),
            "plan_reasoning":            current_plan.get("reasoning"),
        })

        if last_verdict["label"] in ("sufficient", "FULLY_SUFFICIENT", "ANSWER_SUFFICIENT"):
            break

    # ── Answer ───────────────────────────────────────────────────────────────
    if oracle_frozen_result is not None:
        ev_texts = list(oracle_frozen_result["evidence_texts"])
        ev_ids   = list(oracle_frozen_result["evidence_chunk_ids"])
        result = {"answer": oracle_frozen_result["answer"], "evidence": [], "rationale": ""}
    else:
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

        kf_for_answer = all_keyframes
        if answer_keyframe_cap is not None and len(kf_for_answer) > answer_keyframe_cap:
            kf_for_answer = kf_for_answer[:answer_keyframe_cap]

        result = answerer.answer(question, candidates, ev_texts,
                                 keyframe_images=kf_for_answer)
    result["evidence_chunk_ids"] = ev_ids
    result["evidence_texts"]     = ev_texts
    result["trace_iters"]        = iterations
    return result
