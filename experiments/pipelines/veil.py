"""VEIL — iterative evidence retrieval (Subq decomposition + Verifier + Unified Planner).

Per iteration retrieves multiple sub-question queries in parallel:
  iter 0 : sub-question decomposition (LLM-decomposed or option-grounded)
  iter ≥1: Unified Planner — given the verifier's structured signals
           (option_status / failed rubric criteria / missing_evidence / key+distractor ids),
           decides targeted (1-4 queries) vs broadcast (uniform timeline coverage).

Per-iteration retrieval: each sub-question independently fetches BGE coarse top-k
(optionally SigLIP-fused) → optional reranker top-k → de-dup against accumulated
evidence (BGE cosine ≥ ev_dedup_threshold).

Safety nets on iter ≥1 query generation:
  - Jaccard ≥ ``dedup_thresh`` vs prior queries → drop the query.
  - BGE cosine ≥ ``query_drift_threshold`` vs accumulated evidence → drop the query.
  - All targeted queries dropped → fall back to broadcast for this iter.

Call-site variants (``run_experiments.py --pipelines``):
  ``veil_coarse8``  : coarse top-8 only (no reranker)
  ``veil_coarse64`` : coarse top-64 only (no reranker)
  ``veil_rerank8``  : coarse pool → reranker top-8
"""
from __future__ import annotations

import re
from typing import List, Optional, Set

import numpy as np

from experiments.pipelines._keyframes import keyframe_path, load_keyframe_pil
from memory.schema import MemoryBank
from reasoning.verifier import Verifier, get_rubric_dict


# ── Iter-0: sub-question decomposition ────────────────────────────────────────

_SUBQ_SYS = """You are a question decomposer for a video question-answering system.
Decompose the given multiple-choice question into 2-4 atomic sub-questions, each targeting a \
specific piece of information needed to evaluate one or more answer options.

Rules:
- Each sub-question must be self-contained and retrievable from video evidence.
- For "which is NOT correct" / "which is incorrect" questions: generate one sub-question per option.
- For other questions: identify the key atomic facts needed and generate one sub-question per fact.
- Sub-questions must be SHORT (≤15 words), specific, and different from each other.
- Do NOT include option letters in the sub-questions.
- Return ONLY a JSON array of strings, e.g.: ["sub-question 1", "sub-question 2"]"""


def _decompose_into_subquestions(
    question: str,
    candidates: List[str],
    llm,
) -> List[str]:
    """Iter-0: decompose a question into 2-4 atomic sub-questions."""
    import json as _json
    opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Decompose into atomic sub-questions. Return ONLY a JSON array of strings."
    )
    messages = [
        {"role": "system", "content": _SUBQ_SYS},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(messages, max_new_tokens=200, enable_thinking=False)
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            qs = _json.loads(m.group())
            return [str(q).strip() for q in qs if str(q).strip()][:4]
        except Exception:
            pass
    return [question]


def _option_grounded_subquestions(question: str, candidates: List[str]) -> List[str]:
    """Iter-0 alternative: one verification query per answer option (deterministic)."""
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    return [
        (
            f"For option ({letter}) {candidate}: retrieve concrete evidence that supports, "
            f"refutes, or makes unclear whether this option answers the question: {question}"
        )
        for letter, candidate in zip(letters, candidates)
    ]


def _option_grounded_repair_subquestions(
    question: str,
    candidates: List[str],
    missing_description: str,
) -> List[str]:
    """Iter >=1 alternative: one repair query per answer option (deterministic)."""
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    missing = missing_description.strip() or "current evidence is insufficient"
    return [
        (
            f"For option ({letter}) {candidate}: retrieve additional evidence that resolves "
            f"this insufficiency for the question '{question}': {missing}"
        )
        for letter, candidate in zip(letters, candidates)
    ]


# ── Iter ≥1: Unified Planner ─────────────────────────────────────────────────

_PLANNER_UNIFIED_SYS = """You are a retrieval strategy planner for a video question-answering system.
Decide the BEST next retrieval strategy to find evidence for answering the question.

## Strategies

**targeted** — 1-4 focused queries. You decide how many:
  - Use 1 query when the answer requires one specific fact or entity.
  - Use 2-4 queries (sub-questions) when:
    - Answer options name distinct items each needing independent evidence
      (e.g., options are different colors / actions / counts / entities → one query per option)
    - The question involves multiple distinct events or comparisons needing separate retrieval
    - "Which is NOT correct?" → one query per option to verify/falsify each
  Each query must be ≤15 words, self-contained, and target a distinct retrievable fact.

**broadcast** — Uniform sampling across the uncovered video timeline (no query needed).
  Use when:
  - Evidence must cover the WHOLE video (main theme, synopsis, overall narrative)
  - Temporal ordering requires seeing events spread across the full duration
  - Missing analysis indicates broad timeline coverage is needed
  - Targeted search has been tried multiple times without finding sufficient coverage

## Rules
- Do NOT generate queries similar to ones already tried (see plan history).
- For broadcast: set queries to [].

Return ONLY a JSON object:
{
  "strategy": "targeted" | "broadcast",
  "queries": ["..."],
  "reasoning": "<one sentence>"
}"""


def _plan_next(
    question: str,
    candidates: List[str],
    evidence_texts: List[str],
    missing_analysis: str,
    llm,
    plan_history: List[dict] = (),
    allow_broadcast: bool = True,
    option_status: Optional[dict] = None,
    failed_criteria: Optional[dict] = None,
    prune_satisfied: bool = False,
) -> dict:
    """Unified planner for iter ≥1: pick {targeted (n queries), broadcast}.

    Returns ``{"strategy": "targeted"|"broadcast", "queries": [...], "reasoning": "..."}``.
    """
    import json as _json
    opts_lines = []
    for i, c in enumerate(candidates):
        letter = chr(ord('A')+i)
        tag = ""
        if option_status:
            s = option_status.get(letter)
            if s == "verified":   tag = " [VERIFIED]"
            elif s == "excluded": tag = " [EXCLUDED]"
            elif s == "unclear":  tag = " [UNCLEAR]"
            elif s == "conflicting": tag = " [CONFLICTING]"
        opts_lines.append(f"  ({letter}) {c}{tag}")
    opts = "\n".join(opts_lines)

    parts = [f"Question: {question}", f"Options:\n{opts}"]
    if option_status:
        unresolved = [k for k, v in option_status.items() if v in ("unclear", "conflicting")]
        if unresolved:
            parts.append(
                f"NOTE: Options {unresolved} are still unresolved (UNCLEAR or CONFLICTING) — generate queries that target ONLY these options. "
                f"Do NOT generate queries for options already marked VERIFIED or EXCLUDED."
            )
    if failed_criteria:
        lines = [f"  - {name}: {score}" for name, score in failed_criteria.items()]
        parts.append(
            "Failed or weak rubric criteria to repair:\n" + "\n".join(lines) +
            "\nGenerate queries that directly repair these criteria."
        )
    if not allow_broadcast:
        parts.append('NOTE: "broadcast" is NOT allowed for this call. Always output strategy="targeted".')
    if prune_satisfied and plan_history:
        parts.append(
            "NOTE: Review the previously tried queries against the CURRENT evidence. "
            "For each previous query, judge whether its target fact is now adequately covered by the evidence. "
            "Do NOT regenerate queries whose target is already satisfied. "
            "Only generate queries for facts that remain unresolved."
        )

    if plan_history:
        lines = []
        for h in plan_history:
            strat = h.get("strategy", "targeted")
            qs    = h.get("queries") or []
            label = "[broadcast]" if strat == "broadcast" else " | ".join(f'"{q}"' for q in qs)
            lines.append(f"  [{strat}] {label}")
        parts.append("Already tried (do NOT repeat):\n" + "\n".join(lines))

    if evidence_texts:
        covered = _extract_covered_times(evidence_texts)
        if covered:
            parts.append(f"Evidence covers video segments: {covered}")
        ev_sample = "\n".join(f"  [E{i+1}] {t[:200]}" for i, t in enumerate(evidence_texts[:5]))
        parts.append(f"Current evidence (sample):\n{ev_sample}")

    if missing_analysis:
        parts.append(f"Still missing: {missing_analysis}")

    parts.append("Output the retrieval strategy JSON now.")

    messages = [
        {"role": "system", "content": _PLANNER_UNIFIED_SYS},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]
    raw = llm.chat(messages, max_new_tokens=200, enable_thinking=False)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            d        = _json.loads(m.group())
            strategy = str(d.get("strategy", "targeted")).lower()
            if strategy not in ("targeted", "broadcast"):
                strategy = "targeted"
            if not allow_broadcast and strategy == "broadcast":
                strategy = "targeted"
            queries  = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
            if strategy != "broadcast" and not queries:
                strategy = "targeted"
                queries  = [question]
            return {
                "strategy":  strategy,
                "queries":   queries,
                "reasoning": str(d.get("reasoning", "")),
            }
        except Exception:
            pass
    return {"strategy": "targeted", "queries": [question], "reasoning": "parse_failed"}


# ── Verifier → planner repair context ────────────────────────────────────────

def _failed_criteria(criteria: Optional[dict], threshold: float = 1.0) -> dict:
    """Return rubric criteria that are not fully satisfied."""
    failed = {}
    if not isinstance(criteria, dict):
        return failed
    for name, score in criteria.items():
        try:
            val = float(score)
        except (TypeError, ValueError):
            val = 0.0
        if val < threshold:
            failed[str(name)] = val
    return failed


def _planner_repair_context(verdict: dict, use_rubric_judgment: bool) -> tuple[str, dict | None]:
    """Build the only verifier-derived context the planner may use."""
    missing = verdict.get("missing_evidence") or ""
    if isinstance(missing, dict):
        missing = missing.get("description", "") or ""
    missing = str(missing)
    reasoning = str(verdict.get("reasoning") or "")

    if not use_rubric_judgment:
        parts = []
        if reasoning:
            parts.append(f"Verifier reasoning: {reasoning}")
        if missing:
            parts.append(f"Missing evidence: {missing}")
        return "\n".join(parts), None

    failed = _failed_criteria(verdict.get("criteria"))
    status = verdict.get("option_status") or {}
    unresolved = {k: v for k, v in status.items() if v in ("unclear", "conflicting")}

    parts = []
    if failed:
        parts.append("Failed or weak rubric criteria: " +
                     "; ".join(f"{k}={v}" for k, v in failed.items()))
    if unresolved:
        parts.append("Unclear/conflicting options: " +
                     "; ".join(f"{k}={v}" for k, v in unresolved.items()))
    if missing:
        parts.append(f"Missing evidence: {missing}")
    return "\n".join(parts), failed


# ── Broadcast (uniform timeline sampling) ────────────────────────────────────

_BROADCAST_KEYWORDS = frozenset({
    "throughout", "entire video", "whole video", "all segment",
    "timeline", "every segment", "coverage across", "sequence of events",
    "chronolog", "overview of", "synopsis",
})


def _needs_broadcast(missing_desc: str) -> bool:
    s = (missing_desc or "").lower()
    return any(kw in s for kw in _BROADCAST_KEYWORDS)


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


def _extract_covered_times(evidence_texts: List[str]) -> str:
    """Extract time ranges present in evidence texts to inform the planner."""
    pattern = r'\[(\d+)s[-–](\d+)s\]'
    secs = sorted({(int(m.group(1)), int(m.group(2)))
                   for t in evidence_texts
                   for m in re.finditer(pattern, t)})
    def fmt(s: int) -> str:
        return f"{s//60}:{s%60:02d}"
    return ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in secs) if secs else ""


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

    from reasoning.verifier import _format_rubric_as_text
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


# ── Query / chunk de-dup ─────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _too_similar(new_q: str, prev_queries: List[str], threshold: float = 0.5) -> bool:
    return any(_jaccard(new_q, pq) >= threshold for pq in prev_queries)


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


# ── Single-query retrieval primitive ─────────────────────────────────────────

def _retrieve_once(
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
    option_subquestions_each_iter: bool = False,
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
    if force_option_subquestions:
        iter0_queries = _option_grounded_subquestions(question, candidates)
    else:
        iter0_queries = _decompose_into_subquestions(question, candidates, llm)
    current_plan: dict = {
        "strategy":  "targeted",
        "queries":   iter0_queries,
        "reasoning": "iter0_decompose",
    }

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
                t, i, v = _retrieve_once(
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
                                  use_rubric_judgment=use_rubric_judgment)

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
                verdict["missing_evidence"] = None
                verdict["reasoning"] = f"Oracle: pred={_pred} == gold ({verdict.get('reasoning','')})"
            else:
                _missing = _oracle_analyze_missing(
                    question, candidates, all_evidence_texts, gold_letter, _pred, llm)
                verdict["label"] = "insufficient"
                verdict["missing_evidence"] = _missing
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
            "missing_evidence_analysis":  verdict.get("missing_evidence") or None,
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
        m_desc, failed = _planner_repair_context(verdict, use_rubric_judgment=use_rubric_judgment)

        if option_subquestions_each_iter and force_option_subquestions:
            next_plan = {
                "strategy": "targeted",
                "queries": _option_grounded_repair_subquestions(question, candidates, m_desc),
                "reasoning": "option4_repair",
            }
        elif keyword_broadcast and _needs_broadcast(m_desc):
            next_plan = {"strategy": "broadcast", "queries": [], "reasoning": "keyword_trigger"}
        else:
            opt_status = (verdict.get("option_status")
                          if (use_rubric_judgment and use_option_status) else None)
            next_plan = _plan_next(
                question, candidates, all_evidence_texts, m_desc, llm, plan_history,
                allow_broadcast=not keyword_broadcast,
                option_status=opt_status,
                failed_criteria=failed if use_rubric_judgment else None,
                prune_satisfied=prune_satisfied,
            )

        # ── Filter targeted queries; fall back to broadcast if all dropped ───
        if next_plan["strategy"] == "targeted" and not (
            option_subquestions_each_iter and force_option_subquestions
        ):
            ev_mat = (np.array(all_ev_vecs, dtype=np.float32)
                      if query_drift_threshold is not None and all_ev_vecs else None)
            kept: List[str] = []
            for q in next_plan["queries"]:
                if _too_similar(q, prev_queries, dedup_thresh):
                    continue
                if ev_mat is not None:
                    qv    = embedder.encode([q])[0]
                    drift = float(np.max(ev_mat @ qv))
                    if drift >= query_drift_threshold:
                        continue
                kept.append(q)
            if kept:
                next_plan["queries"] = kept
            else:
                next_plan = {"strategy": "broadcast", "queries": [],
                             "reasoning": "all_targeted_filtered"}

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
