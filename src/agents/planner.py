"""Planner — generate retrieval queries for the VEIL iterative loop.

Two-stage planning:
  iter-0: sub-question decomposition (LLM-decomposed or option-grounded).
  iter≥1: unified planner — given verifier signals (unknown_options / failed rubric
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

_SUBQ_SYS = """\
You are a retrieval planner for multiple-choice long-video question answering.

Your task: generate retrieval subqueries that look for evidence in the video
memory for or against each option.

You must not answer the question.

Rules:
1. For each option (or set of related options), generate ONE search query
   that targets the evidence needed to verify or refute it.
2. Prefer queries that DISTINGUISH confusing options — focus on what differs
   across options, not what they share.
3. Use concrete keywords from the question and the options
   (distinctive nouns, numbers, names, actions).
4. Each query is one short sentence (≤ 15 words), self-contained,
   without option letters (A/B/C/D).
5. Do not invent details not in the question or options.
6. Generate AT MOST 4 subqueries. If options share most content,
   focus only on the distinguishing detail.

Output valid JSON only.

Output format (extend the arrays to cover ALL options, up to 4 subqueries):
{
  "question_focus": "<what evidence is needed to choose among options>",
  "option_claims": [
    {"option": "A", "claim": "<verifiable claim implied by option A>"},
    {"option": "B", "claim": "<verifiable claim implied by option B>"},
    {"option": "C", "claim": "<verifiable claim implied by option C>"},
    {"option": "D", "claim": "<verifiable claim implied by option D>"}
  ],
  "subqueries": [
    {"id": "q1", "related_options": ["A"], "query": "<query targeting A>"},
    {"id": "q2", "related_options": ["B"], "query": "<query targeting B>"},
    {"id": "q3", "related_options": ["C", "D"], "query": "<shared distinguishing query>"}
  ]
}"""


def _decompose_into_subquestions(
    question: str,
    candidates: List[str],
    llm,
) -> List[str]:
    """Iter-0: decompose a question into ≤4 retrieval subqueries.

    Expects the LLM to return a JSON object with a ``subqueries`` field
    (option-grounded, evidence-typed); extracts the ``query`` string from each.
    Falls back to the original question on parse failure.
    """
    opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
    user = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Generate the retrieval plan as JSON now. Output only the JSON object."
    )
    messages = [
        {"role": "system", "content": _SUBQ_SYS},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(messages, max_new_tokens=500, enable_thinking=False)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group())
            subs = obj.get("subqueries") or []
            queries = [str(s.get("query", "")).strip() for s in subs
                       if isinstance(s, dict) and str(s.get("query", "")).strip()]
            if queries:
                return queries[:4]
        except Exception:
            pass
    return [question]


def _option_grounded_subquestions(question: str, candidates: List[str]) -> List[str]:
    """Iter-0 alternative: one verification query per answer option."""
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    return [
        (
            f"For option ({letter}) {candidate}: retrieve concrete evidence that supports, "
            f"refutes, or makes unclear whether this option answers the question: {question}"
        )
        for letter, candidate in zip(letters, candidates)
    ]


# ── Iter ≥1: Unified Planner ─────────────────────────────────────────────────

_PLANNER_UNIFIED_SYS = """\
You are a retrieval planner for iterative video question answering.

Round-1+ context: A retrieval round has been completed. The user message
gives you:
- The question and options (some options may be tagged [UNKNOWN] — the
  verifier could not yet judge them from current evidence)
- Evidence already retrieved (and the time segments it covers)
- Queries already tried in previous rounds
- Missing evidence analysis from the verifier
- Optionally, weak rubric criteria that still need supporting evidence

Decide between two strategies:

**targeted** — Issue 1–4 new queries to fill specific gaps.
  Each query MUST:
  - Aim at a specific [UNKNOWN] option, a specific weak rubric criterion,
    or a specific missing fact named in the verifier's analysis.
  - Be NEW — do not repeat or paraphrase a previously-tried query.
  - Be ≤ 15 words, atomic, retrieval-friendly.
  - Use concrete keywords (distinctive nouns, numbers, names, actions);
    do NOT include option letters (A/B/C/D).

**broadcast** — Sample frames uniformly across uncovered video timeline.
  Choose this when:
  - The question needs full-video coverage (synopsis, overall theme,
    full-video sequence of events, "throughout the video" reasoning).
  - Previous rounds' targeted queries have repeatedly failed to surface
    new useful evidence.
  - There is no specific entity / fact left to search for, only a general
    "what else is there" need.

Do not answer the question; only plan retrieval.

Output valid JSON only:
{
  "strategy": "targeted" | "broadcast",
  "queries": ["...", "..."],
  "reasoning": "<one sentence: which gap this targets, OR why broadcast>"
}
For broadcast, set queries to []."""


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
    plan_history: List[List[str]] = (),
    unknown_options: Optional[List[str]] = None,
    weak_rubric_criteria: Optional[dict] = None,
    prune_satisfied: bool = False,
    global_context: str = "",
) -> dict:
    """Unified planner for iter ≥1: pick {targeted (n queries), broadcast}.

    Returns ``{"strategy": "targeted"|"broadcast", "queries": [...], "reasoning": "..."}``.
    """
    unknown_set = set(unknown_options or [])
    opts_lines = []
    for i, c in enumerate(candidates):
        letter = chr(ord('A')+i)
        tag = " [UNKNOWN]" if letter in unknown_set else ""
        opts_lines.append(f"  ({letter}) {c}{tag}")
    opts = "\n".join(opts_lines)

    parts = []
    if global_context:
        parts.append(f"[Video Context — coarse overview to guide query generation]\n{global_context}")
    parts += [f"Question: {question}", f"Options:\n{opts}"]
    if unknown_options:
        parts.append(f"Current unresolved options: {unknown_options}.")
    if weak_rubric_criteria:
        lines = [f"  - {name}" for name in weak_rubric_criteria]
        parts.append(
            "Weak rubric criteria to repair:\n" + "\n".join(lines) +
            "\nGenerate queries that directly address these criteria."
        )
    if prune_satisfied and plan_history:
        parts.append(
            "NOTE: Review the previously tried queries against the CURRENT evidence. "
            "For each previous query, judge whether its target fact is now adequately covered by the evidence. "
            "Do NOT regenerate queries whose target is already satisfied. "
            "Only generate queries for facts that remain unresolved."
        )

    if plan_history:
        lines = []
        for qs in plan_history:
            if qs:
                lines.append("  " + " | ".join(f'"{q}"' for q in qs))
            else:
                lines.append("  [broadcast]")
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

def _planner_repair_context(verdict: dict, rubric_judgment: bool) -> Tuple[str, Optional[List[str]]]:
    """Build verifier-derived context for the planner."""
    missing = str(verdict.get("missing_evidence_analysis") or "").strip()

    if not rubric_judgment:
        return (f"Missing evidence analysis: {missing}" if missing else ""), None

    weak_list: List[str] = list(verdict.get("weak_rubric_criteria") or [])
    unknown = list(verdict.get("unknown_options") or [])

    parts = []
    if weak_list:
        parts.append("Weak rubric criteria: " + ", ".join(weak_list))
    if unknown:
        parts.append("Unresolved options: " + ", ".join(unknown))
    if missing:
        parts.append(f"Missing evidence analysis: {missing}")
    return "\n".join(parts), weak_list or None


# ── Query similarity safety nets ─────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _too_similar(new_q: str, prev_queries: List[str], threshold: float = 0.9) -> bool:
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
        """Iter-0 plan.

        Two modes:
          - force_option=True  -> one option-grounded query per candidate
          - force_option=False -> LLM splits into 2-4 atomic sub-queries
        """
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
        rubric_judgment: bool,
    ) -> Tuple[str, Optional[List[str]]]:
        """Project a verifier verdict into (missing_description, weak_rubric_criteria)."""
        return _planner_repair_context(verdict, rubric_judgment)

    def plan_next(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        verdict: dict,
        plan_history: List[List[str]],
        *,
        rubric_judgment: bool = True,
        prune_satisfied: bool = False,
        global_context: str = "",
    ) -> Tuple[dict, str, Optional[List[str]]]:
        """Pick the next iter plan from the verifier verdict.

        Returns (plan, missing_description, weak_rubric_criteria).
        global_context: optional coarse video overview (e.g. top-2 L3 summaries) prepended to prompt.
        """
        m_desc, weak = _planner_repair_context(verdict, rubric_judgment)

        unk_opts = verdict.get("unknown_options") if rubric_judgment else None
        plan = _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            unknown_options=unk_opts,
            weak_rubric_criteria=weak if rubric_judgment else None,
            prune_satisfied=prune_satisfied,
            global_context=global_context,
        )
        return plan, m_desc, weak

    def filter_targeted_queries(
        self,
        plan: dict,
        plan_history: List[List[str]],
        embedder,
        evidence_vecs: List[List[float]],
        *,
        dedup_thresh: float = 0.9,
        drift_threshold: Optional[float] = 0.70,
    ) -> dict:
        """Apply Jaccard + BGE-drift safety nets to a targeted plan.

        Falls back to broadcast if all targeted queries are dropped.
        """
        if plan["strategy"] != "targeted":
            return plan

        prev_queries = [q for qs in plan_history for q in qs]
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
