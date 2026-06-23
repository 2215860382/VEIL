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

_PLANNER_LEGACY_SYS = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence and what is still missing (verifier-supplied weak rubric
criteria + per-option diagnoses), choose the next retrieval strategy.

## Strategies

**targeted** — Issue 1–4 focused BGE queries to retrieve specific missing evidence.
  Default. Each query targets a specific failing diagnosis and is distinct
  from previously tried queries.

**broadcast** — Sample frames uniformly across the uncovered video timeline (no queries).
  Use for whole-video coverage (main theme / synopsis, events spread across the
  full duration) or when targeted queries have repeatedly failed.

## How to write queries
Read the per-option `diagnosis` lines under each weak criterion. They name the
EXACT missing fact (e.g. "option A lacks the name of the award presenter").
Write each query so it would retrieve THAT fact — concrete entity names, event
descriptions, or time anchors. Do not paraphrase the original question.

## Output format
Return ONLY a JSON object:
{
  "strategy": "targeted" | "broadcast",
  "queries": ["..."],
  "reasoning": "<one sentence>"
}
For broadcast, set queries to []."""


_PLANNER_UNIFIED_SYS = """You are a retrieval strategy planner for a video question-answering system.
Given the current evidence, what is still missing, and the verifier's diagnostic
hints (weak rubric criteria with their suggested repair actions), choose the BEST
next strategy.

## Strategies

**targeted** — Issue 1–4 focused BGE queries to retrieve specific missing evidence.
  Default when the verifier signal points to "refine_query". Each query is
  self-contained and does not repeat what was already tried.

**broadcast** — Sample frames uniformly across the uncovered video timeline (no queries).
  Use for whole-video coverage: main theme / synopsis, events spread across the
  full duration, or when targeted queries have repeatedly failed.

**asr_match** — Substring/phrase match against chunk ASR (dialogue) text.
  Use when the QUESTION quotes or paraphrases a specific spoken phrase
  (e.g. "When X says '...'") and the verifier flagged `quote_grounding`. Set
  `queries` to the exact quoted phrase(s) to locate in the ASR.

**time_sorted** — Reorder the already-retrieved evidence by chunk start_time
  (no new retrieval). Use when the verifier flagged `chronological_anchor` or
  `temporal_consistency` weak — evidence content is there but order is wrong.

**dense_sample** — Pull additional keyframes from chunks already in evidence.
  Use when the verifier flagged `temporal_alignment`, `counting_visual`,
  `visual_observability`, `action_visibility`, or `transition_process_coverage` —
  the right chunk is found but its 1-frame sample is too sparse for the question.

**loose_verify** — Tell the next verifier pass to accept indirect / inferred
  evidence (no new retrieval). Use when the verifier flagged
  `implicit_causal` / `causal_link_completeness` — evidence supports inference
  but verifier is being too strict about explicit statements.

## How to pick a strategy
Look at `failure_repair_action` on the weakest weak_rubric_criteria entries.
Map each action to the corresponding strategy of the same name. If multiple
weak criteria suggest conflicting actions, pick the action of the SINGLE
lowest-scoring criterion. Fall back to `targeted` when no weak criteria.

## Output format
Return ONLY a JSON object:
{
  "strategy": "targeted" | "broadcast" | "asr_match" | "time_sorted" | "dense_sample" | "loose_verify",
  "queries": ["..."],
  "reasoning": "<one sentence explaining the choice>"
}
For broadcast / time_sorted / dense_sample / loose_verify, set queries to []."""


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
    legacy_only: bool = False,
) -> dict:
    """Unified planner for iter ≥1: pick {targeted (n queries), broadcast}.

    Returns ``{"strategy": "targeted"|"broadcast", "queries": [...], "reasoning": "..."}``.

    legacy_only=True restricts the strategy space to {targeted, broadcast} and
    suppresses ``failure_repair_action`` hints in the prompt. Used to isolate the
    "signal repair" effect from the action-space expansion in ablation runs.
    """
    unknown_set = set(unknown_options or [])
    opts_lines = []
    for i, c in enumerate(candidates):
        letter = chr(ord('A')+i)
        tag = " [UNKNOWN]" if letter in unknown_set else ""
        opts_lines.append(f"  ({letter}) {c}{tag}")
    opts = "\n".join(opts_lines)

    parts = [f"Question: {question}", f"Options:\n{opts}"]
    if unknown_options:
        parts.append(f"Current unresolved options: {unknown_options}.")
    if weak_rubric_criteria:
        # Accept both legacy List[str] and new List[Dict] formats.
        lines = []
        actions_seen = []
        for item in weak_rubric_criteria:
            if isinstance(item, dict):
                name   = item.get("name", "")
                score  = item.get("score")
                action = item.get("failure_repair_action", "refine_query")
                desc   = item.get("description", "")
                failing = item.get("failing_options") or []
                diagnosis = item.get("diagnosis") or {}
                actions_seen.append(action)
                bits = [f"  - {name}"]
                if score is not None:
                    bits.append(f"(score={score})")
                if not legacy_only:
                    bits.append(f"[suggested action: {action}]")
                if failing:
                    bits.append(f"failing on options {failing}")
                if desc:
                    bits.append(f"— {desc}")
                lines.append(" ".join(bits))
                # Per-option diagnosis: the verifier's concrete reason this
                # criterion fails for each failing option. Use these to write
                # targeted queries instead of paraphrasing the question.
                if isinstance(diagnosis, dict):
                    for opt in failing or sorted(diagnosis.keys()):
                        d = diagnosis.get(opt)
                        if isinstance(d, str) and d.strip():
                            lines.append(f"      · option {opt}: {d.strip()}")
            else:
                lines.append(f"  - {item}")
        parts.append(
            "Weak rubric criteria to repair (weakest first):\n" + "\n".join(lines)
        )
        if actions_seen and not legacy_only:
            top_action = actions_seen[0]  # weakest criterion's action
            parts.append(
                f"The weakest criterion suggests action `{top_action}` — "
                "strongly prefer the corresponding strategy."
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

    sys_prompt = _PLANNER_LEGACY_SYS if legacy_only else _PLANNER_UNIFIED_SYS
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]
    raw = llm.chat(messages, max_new_tokens=240, enable_thinking=False)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if legacy_only:
        valid_strategies = {"targeted", "broadcast"}
        needs_queries    = {"targeted"}
    else:
        valid_strategies = {
            "targeted", "broadcast", "asr_match", "time_sorted",
            "dense_sample", "loose_verify",
        }
        needs_queries = {"targeted", "asr_match"}
    if m:
        try:
            d        = _json.loads(m.group())
            strategy = str(d.get("strategy", "targeted")).lower()
            if strategy not in valid_strategies:
                strategy = "targeted"
            queries  = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
            if strategy in needs_queries and not queries:
                # asr_match / targeted without queries is useless; degrade to targeted+question
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

    weak_list = list(verdict.get("weak_rubric_criteria") or [])
    unknown = list(verdict.get("unknown_options") or [])

    parts = []
    if weak_list:
        # Render either legacy List[str] or new List[Dict] (with action+desc).
        names_only = []
        for w in weak_list:
            if isinstance(w, dict):
                names_only.append(str(w.get("name", "")))
            else:
                names_only.append(str(w))
        parts.append("Weak rubric criteria: " + ", ".join(names_only))
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
        legacy_only: bool = False,
    ) -> Tuple[dict, str, Optional[List[str]]]:
        """Pick the next iter plan from the verifier verdict.

        Returns (plan, missing_description, weak_rubric_criteria).

        legacy_only=True restricts strategies to {targeted, broadcast} —
        used to isolate the verifier signal-repair effect from the planner
        action-space expansion in ablation runs.
        """
        m_desc, weak = _planner_repair_context(verdict, rubric_judgment)

        unk_opts = verdict.get("unknown_options") if rubric_judgment else None
        plan = _plan_next(
            question, candidates, evidence_texts, m_desc, self.llm, plan_history,
            unknown_options=unk_opts,
            weak_rubric_criteria=weak if rubric_judgment else None,
            prune_satisfied=prune_satisfied,
            legacy_only=legacy_only,
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
        # Only `targeted` goes through Jaccard+drift filters. `asr_match`
        # uses exact phrases (dedup would over-match), and the other three
        # (broadcast / time_sorted / dense_sample / loose_verify) carry no
        # queries.
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
