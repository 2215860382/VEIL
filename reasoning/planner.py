"""Planner — generate retrieval queries for the VEIL iterative loop.

Two-stage planning:
  iter-0: sub-question decomposition (LLM-decomposed or option-grounded).
  iter≥1: unified planner — given verifier signals (option_status / failed rubric
          criteria / missing_evidence_analysis), pick {targeted (1-4 queries),
          broadcast (uniform timeline coverage)}.

Safety nets on iter≥1 query generation (applied via ``filter_targeted_queries``):
  - Jaccard ≥ ``dedup_thresh`` vs prior queries → drop the query.
  - BGE cosine ≥ ``drift_threshold`` vs accumulated evidence → drop the query.
  - All targeted queries dropped → fall back to broadcast for this iter.
"""
from __future__ import annotations

import json as _json
import re
from typing import List, Optional, Tuple

import numpy as np


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


# ── Iter ≥1: Unified Planner ─────────────────────────────────────────────────

_PLANNER_UNIFIED_SYS = """You are a retrieval strategy planner for a video question-answering system.
Decide the BEST next retrieval strategy to find evidence for answering the question.

## Strategies

**targeted** — 1-4 focused queries. You decide how many:
  - Use 1 query when the answer requires one specific fact, comparison, time relation, or entity.
  - Use 2-4 queries (sub-questions) when:
    - Multiple answer options are still unresolved and truly need independent evidence
      (e.g., different colors / actions / counts / entities)
    - The question involves multiple distinct events or comparisons needing separate retrieval
    - "Which is NOT correct?" and several options are still unresolved
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


def _extract_covered_times(evidence_texts: List[str]) -> str:
    """Extract time ranges present in evidence texts to inform the planner."""
    pattern = r'\[(\d+)s[-–](\d+)s\]'
    secs = sorted({(int(m.group(1)), int(m.group(2)))
                   for t in evidence_texts
                   for m in re.finditer(pattern, t)})
    def fmt(s: int) -> str:
        return f"{s//60}:{s%60:02d}"
    return ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in secs) if secs else ""


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
                f"Current unresolved options: {unresolved}."
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


def _planner_repair_context(verdict: dict, use_rubric_judgment: bool) -> Tuple[str, Optional[dict]]:
    """Build the only verifier-derived context the planner may use."""
    missing = verdict.get("missing_evidence_analysis") or ""
    missing_lines: List[str] = []
    if isinstance(missing, dict):
        focus_options = missing.get("focus_options") or []
        if focus_options:
            missing_lines.append("Focus options from missing analysis: " + ", ".join(map(str, focus_options)))
        analysis = str(missing.get("analysis") or "").strip()
        if analysis:
            missing_lines.append("Missing analysis: " + analysis)
        time_scope = str(missing.get("time_scope") or "").strip()
        if time_scope:
            missing_lines.append(f"Time scope from missing analysis: {time_scope}")
        conflict_fact = str(missing.get("conflict_fact") or "").strip()
        if conflict_fact:
            missing_lines.append(f"Conflict fact from missing analysis: {conflict_fact}")
        missing = "\n".join(missing_lines)
    else:
        missing = str(missing)

    if not use_rubric_judgment:
        parts = []
        if missing:
            parts.append(f"Missing evidence analysis: {missing}")
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
        parts.append(f"Missing evidence analysis: {missing}")
    return "\n".join(parts), failed


# ── Broadcast keyword trigger ────────────────────────────────────────────────

_BROADCAST_KEYWORDS = frozenset({
    "throughout", "entire video", "whole video", "all segment",
    "timeline", "every segment", "coverage across", "sequence of events",
    "chronolog", "overview of", "synopsis",
})


def _needs_broadcast(missing_desc: str) -> bool:
    s = (missing_desc or "").lower()
    return any(kw in s for kw in _BROADCAST_KEYWORDS)


# ── Query similarity safety nets ─────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _too_similar(new_q: str, prev_queries: List[str], threshold: float = 0.5) -> bool:
    return any(_jaccard(new_q, pq) >= threshold for pq in prev_queries)


# ── Public class ─────────────────────────────────────────────────────────────

class Planner:
    """Encapsulates iter-0 decomposition + iter≥1 unified planner + query filtering.

    Holds only the LLM client; embedder and evidence vectors are passed per-call
    so the Planner stays parallel to ``Verifier`` (one external service only).
    """

    def __init__(self, llm):
        self.llm = llm

    def decompose_iter0(
        self,
        question: str,
        candidates: List[str],
        *,
        force_option: bool = False,
    ) -> dict:
        """Iter-0 plan: LLM atomic-fact decomposition (default) or per-option queries."""
        if force_option:
            return {
                "strategy":  "targeted",
                "queries":   _option_grounded_subquestions(question, candidates),
                "reasoning": "iter0_option_grounded",
            }
        return {
            "strategy":  "targeted",
            "queries":   _decompose_into_subquestions(question, candidates, self.llm),
            "reasoning": "iter0_decompose",
        }

    def repair_context(
        self,
        verdict: dict,
        use_rubric_judgment: bool,
    ) -> Tuple[str, Optional[dict]]:
        """Project a verifier verdict into (missing_description, failed_criteria)."""
        return _planner_repair_context(verdict, use_rubric_judgment)

    def plan_next(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        verdict: dict,
        plan_history: List[dict],
        *,
        keyword_broadcast: bool = True,
        use_rubric_judgment: bool = True,
        use_option_status: bool = True,
        prune_satisfied: bool = False,
    ) -> Tuple[dict, str, Optional[dict]]:
        """Pick the next iter plan from the verifier verdict.

        Returns (plan, missing_description, failed_criteria).
        The latter two are also returned so callers can record them in the trace.
        """
        m_desc, failed = _planner_repair_context(verdict, use_rubric_judgment)

        if keyword_broadcast and _needs_broadcast(m_desc):
            plan = {"strategy": "broadcast", "queries": [], "reasoning": "keyword_trigger"}
            return plan, m_desc, failed

        opt_status = (verdict.get("option_status")
                      if (use_rubric_judgment and use_option_status) else None)
        plan = _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            allow_broadcast=not keyword_broadcast,
            option_status=opt_status,
            failed_criteria=failed if use_rubric_judgment else None,
            prune_satisfied=prune_satisfied,
        )
        return plan, m_desc, failed

    def filter_targeted_queries(
        self,
        plan: dict,
        prev_queries: List[str],
        embedder,
        evidence_vecs: List[List[float]],
        *,
        dedup_thresh: float = 0.5,
        drift_threshold: Optional[float] = 0.70,
    ) -> dict:
        """Apply Jaccard + BGE-drift safety nets to a targeted plan.

        Falls back to broadcast if all targeted queries are dropped.
        """
        if plan["strategy"] != "targeted":
            return plan

        ev_mat = (np.array(evidence_vecs, dtype=np.float32)
                  if drift_threshold is not None and evidence_vecs else None)
        kept: List[str] = []
        for q in plan["queries"]:
            if _too_similar(q, prev_queries, dedup_thresh):
                continue
            if ev_mat is not None:
                qv    = embedder.encode([q])[0]
                drift = float(np.max(ev_mat @ qv))
                if drift >= drift_threshold:
                    continue
            kept.append(q)
        if kept:
            return {**plan, "queries": kept}
        return {"strategy": "broadcast", "queries": [],
                "reasoning": "all_targeted_filtered"}
