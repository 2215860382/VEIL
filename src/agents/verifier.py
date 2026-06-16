"""Verifier — judge whether retrieved evidence satisfies the rubric for this question type.

Rubric-guided judgment:
  1. Evidence attribution – per-evidence, per-option support / refute / neutral / conflict.
  2. Option status        – verified / excluded / unclear / conflicting for each option.
  3. Rubric criteria      – explicit 0 / 0.5 / 1 per criterion; aggregated to a score.
  4. Label and gaps       – label follows the rubric threshold; gaps drive the next query.

Rubric lives in ``rubric_templates.yaml`` (structured YAML, pluggable per task_type).
Currently all questions use ``templates.default``; extend ``type_aliases`` / ``keyword_rules``
in the yaml to route different types to different templates without code changes.
"""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.extract_json import as_str, extract_json


def _inject_images(message: dict, pil_images) -> dict:
    """Replace str content with [image_url…, text] for multimodal API calls."""
    import base64, io
    valid = [img for img in pil_images if img is not None]
    if not valid:
        return message
    content = []
    for img in valid:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": message["content"]})
    return {"role": message["role"], "content": content}


# ── YAML loading ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _rubric_config() -> Tuple[Dict, Dict, Dict[str, str], List[Tuple[List[str], str]]]:
    path = Path(__file__).with_name("rubric_templates.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"VEIL rubric templates missing: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    general: Dict = data.get("general") or {}
    if not general.get("rubric_criteria"):
        raise ValueError("rubric_templates.yaml must define general.rubric_criteria")

    templates: Dict = {}
    for k, v in (data.get("templates") or {}).items():
        templates[k] = v

    aliases: Dict[str, str] = dict(data.get("type_aliases") or {})
    rules_raw = data.get("keyword_rules") or []
    keyword_rules: List[Tuple[List[str], str]] = []
    for row in rules_raw:
        kws = row.get("keywords") or []
        qtype = row.get("qtype", "")
        if isinstance(kws, str):
            kws = [kws]
        keyword_rules.append((list(kws), str(qtype)))

    return general, templates, aliases, keyword_rules


def get_rubric_dict(question: str, task_type: Optional[str] = None) -> dict:
    """Return the combined rubric dict for the given question / task type.

    Always includes general criteria. If a type-specific template matches,
    its criteria are appended and its scoring_rule / sufficient_threshold apply.
    Falls back to general's scoring_rule / sufficient_threshold when no type matches.
    """
    general, templates, aliases, keyword_rules = _rubric_config()

    # Resolve type-specific template key
    key: Optional[str] = None
    if task_type:
        key = aliases.get(task_type)
        if key not in templates:
            key = None

    # keyword_rules override (question-text based routing)
    q_lower = (question or "").lower()
    for kws, qtype in keyword_rules:
        if any(kw in q_lower for kw in kws) and qtype in templates:
            key = qtype
            break

    if key is not None:
        tpl = templates[key]
        return {
            "rubric_criteria": general["rubric_criteria"] + (tpl.get("rubric_criteria") or []),
            "scoring_rule":         tpl.get("scoring_rule",         general.get("scoring_rule", "average")),
            "sufficient_threshold": tpl.get("sufficient_threshold", general.get("sufficient_threshold", 0.75)),
        }
    # No type match: general only
    return dict(general)


def get_rubric(question: str, task_type: Optional[str] = None) -> str:
    """Return the rubric as a formatted text string (backward-compat helper)."""
    d = get_rubric_dict(question, task_type)
    if "_legacy_text" in d:
        return d["_legacy_text"]
    return _format_rubric_as_text(d)


def _format_rubric_as_text(d: dict) -> str:
    lines = []
    for crit in d.get("rubric_criteria") or []:
        weight = crit.get("weight", 1.0)
        wtxt = f" [weight={weight}]" if float(weight) != 1.0 else ""
        lines.append(
            f"  Criterion [{crit['name']}] {wtxt}: {crit['description']} "
            f"(1={crit['score_1']}; 0.5={crit['score_half']}; 0={crit['score_0']})"
        )
    rule = d.get("scoring_rule", "average")
    thr  = d.get("sufficient_threshold", 0.5)
    lines.append(f"  Scoring: {rule} of criteria; sufficient if score >= {thr}")
    return "\n".join(lines)


# ── Verifier prompt ────────────────────────────────────────────────────────────

VERIFIER_SYS = """\
You evaluate whether retrieved video evidence is sufficient to judge each answer option.
Output ONE strict JSON object — no prose, no markdown fences.

## CORE PRINCIPLE — read carefully (this overrides common-sense scoring)
You are NOT asked "does the evidence support option X?"
You are asked "does the evidence ALLOW A JUDGMENT on option X?"

Evidence that CLEARLY RULES OUT an option scores AS HIGH as evidence that CLEARLY CONFIRMS it.
Both confirm and refute count as "sufficient to judge". Only return low scores when the
evidence is genuinely silent or ambiguous about that criterion.

Examples:
- Q: "What color is the car?" Option: "Red car". Evidence: "The car is blue."
  → evidence_coverage = 1.0 (we CAN judge: option is FALSE)
- Q: "How many cats?" Option: "Three cats". Evidence: "Two cats sit on the sofa."
  → evidence_coverage = 1.0 (we CAN judge: option is FALSE — two, not three)
- Q: "What is the brand?" Option: "Sony". Evidence: "A modern TV is shown."
  → evidence_coverage = 0.0 (evidence silent on brand; cannot judge)

## Step 1 — Per-Option Evidence Sufficiency Scoring
For each option × each rubric criterion (use the EXACT criterion names from the Rubric below):
  1.0 = evidence allows a clear judgment on this criterion (confirm OR rule out).
  0.5 = some relevant signal but truly ambiguous.
  0.0 = evidence is silent or irrelevant for this criterion.
When unsure between 1.0 and 0.5, choose 0.5; between 0.5 and 0.0, choose 0.0.

## Step 2 — Option Judgment
Based on the evidence as a whole:
  true    : evidence (confirm or strong implication) makes this option correct
  false   : evidence (refute or contradiction) makes this option incorrect
  unknown : evidence cannot establish either direction

## Step 3 — Missing Evidence Analysis (structured)
For options that ended up "unknown", describe what specific evidence is still missing.
Leave fields null / empty list if no option is unknown.

## Output schema — return ONLY this JSON
{
  "option_criteria_scores": {
    "A": {"evidence_coverage": 1.0, "evidence_consistency": 0.5, "<other_criterion>": 0.0},
    "B": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0},
    "C": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0},
    "D": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0}
  },
  "option_judgment": {
    "A": "true",
    "B": "false",
    "C": "unknown",
    "D": "unknown"
  },
  "missing_evidence_analysis": {
    "focus_options": ["C", "D"],
    "target_fact": "What the speaker says about ingredient counts after the second cut.",
    "time_scope": "120s-180s",
    "conflict": null
  }
}

Rules for the JSON:
- Criterion names in option_criteria_scores MUST match the Rubric names EXACTLY
  (lowercase snake_case as written under [name] in the Rubric block).
- Use the OPTION LETTERS that actually appear in the Options block.
- target_fact: ONE actionable sentence naming the specific fact still needed.
- time_scope: a range like "120s-180s" if evidence should come from a specific
  segment, otherwise null.
- conflict: name an unresolved contradiction across evidence if any, otherwise null.
- focus_options = [] and other fields = null when all options are true/false."""

# Looser variant — explicitly allows synthesizing indirect / partial evidence
# instead of requiring verbatim restatement of the option.
VERIFIER_SYS_LOOSE = """\
You evaluate whether retrieved video evidence is sufficient to judge each answer option.
Output ONE strict JSON object — no prose, no markdown fences.

## CORE PRINCIPLE — read carefully (this overrides common-sense scoring)
You are NOT asked "does the evidence support option X?"
You are asked "does the evidence ALLOW A JUDGMENT on option X?"

Evidence that CLEARLY RULES OUT an option scores AS HIGH as evidence that CLEARLY CONFIRMS it.
Both confirm and refute count as "sufficient to judge".

LOOSE MODE — accept indirect / inferred evidence:
Video memory is a high-level summary; it almost never restates an answer option verbatim.
Treat evidence as INDIRECT:
- Multiple pieces that together IMPLY the option count as sufficient to judge.
- Paraphrased speech / subtitle quotes count as explicit statements.
- Visual descriptions (objects, actions, locations) count as judgable evidence
  for perception / counting / spatial / temporal-order claims.
- Only mark "silent" when the evidence is truly off-topic — NOT just because
  it lacks word-for-word phrasing of the option.

Examples:
- Q: "What color is the car?" Option: "Red car". Evidence: "The car is blue."
  → evidence_coverage = 1.0 (CAN judge: option is FALSE)
- Q: "Why did the price drop?" Option: "Supply increase". Evidence: "$10 → $4000
  over six months as more sellers entered the market."
  → evidence_coverage = 1.0 (CAN judge by inference: option is TRUE)

## Step 1 — Per-Option Evidence Sufficiency Scoring
For each option × each rubric criterion (use the EXACT criterion names from the Rubric below):
  1.0 = evidence allows a clear judgment (confirm OR rule out), including via
        reasonable inference from indirect evidence.
  0.5 = some relevant signal but truly ambiguous.
  0.0 = no usable signal at all for this criterion.

## Step 2 — Option Judgment
Prefer true / false over unknown whenever the evidence permits a reasonable inference.
  true    : evidence (direct or by inference) establishes the option is correct
  false   : evidence (refute or contradiction) establishes the option is incorrect
  unknown : evidence is genuinely silent on this option

## Step 3 — Missing Evidence Analysis (structured)
For options that ended up "unknown", describe what specific evidence is still missing.
Leave fields null / empty list if no option is unknown.

## Output schema — return ONLY this JSON
{
  "option_criteria_scores": {
    "A": {"evidence_coverage": 1.0, "evidence_consistency": 0.5, "<other_criterion>": 0.0},
    "B": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0},
    "C": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0},
    "D": {"evidence_coverage": 0.0, "evidence_consistency": 0.0, "<other_criterion>": 0.0}
  },
  "option_judgment": {
    "A": "true",
    "B": "false",
    "C": "unknown",
    "D": "unknown"
  },
  "missing_evidence_analysis": {
    "focus_options": ["C", "D"],
    "target_fact": "What the speaker says about ingredient counts after the second cut.",
    "time_scope": "120s-180s",
    "conflict": null
  }
}

Rules for the JSON:
- Criterion names in option_criteria_scores MUST match the Rubric names EXACTLY
  (lowercase snake_case as written under [name] in the Rubric block).
- Use the OPTION LETTERS that actually appear in the Options block.
- target_fact: ONE actionable sentence naming the specific fact still needed.
- time_scope: a range like "120s-180s" if evidence should come from a specific
  segment, otherwise null.
- conflict: name an unresolved contradiction across evidence if any, otherwise null.
- focus_options = [] and other fields = null when all options are true/false."""


VERIFIER_SYS_NORUBRIC = """\
You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.
This is no-rubric mode:
  - Do NOT use any rubric.
  - Do NOT perform evidence attribution.
  - Do NOT produce per-option status.
Judge the historical evidence chain holistically, decide whether it is enough to answer, and explain briefly.

## Step 1 — Holistic Sufficiency Judgment
Look at the evidence chain overall and decide whether it is enough to confidently pick an answer.
  • "sufficient"   if you are confident the evidence supports a specific option
  • "insufficient" otherwise

## Step 2 — Reasoning & Missing Evidence
Write 1-2 sentences explaining your overall judgment.
If insufficient, output a SEMI-STRUCTURED "missing_evidence_analysis" object with:
  - "focus_options": []
  - "analysis": one dense, actionable sentence or short paragraph describing what evidence is still needed next
  - "time_scope": concrete time range if needed, otherwise null
  - "conflict_fact": exact conflicting fact if any, otherwise null

Return ONLY this JSON:
{
  "criteria":         {},
  "score":            0.0,
  "reasoning":        "...",
  "label":            "sufficient" or "insufficient",
  "missing_evidence_analysis": {
    "focus_options": [],
    "analysis": "Need direct evidence establishing whether event X happened before or after event Y.",
    "time_scope": null,
    "conflict_fact": null
  }
}"""


def _format_rubric_for_user(rubric: dict) -> str:
    """Format rubric as a user-message section (criteria + scoring rule)."""
    crits = rubric.get("rubric_criteria") or []
    rule  = rubric.get("scoring_rule", "average")
    thr   = rubric.get("sufficient_threshold", 0.5)

    lines = [f"### Rubric Criteria  (scoring: {rule} of scores; sufficient if score ≥ {thr})"]
    for c in crits:
        weight = c.get("weight", 1.0)
        weight_suffix = f" [weight={weight}]" if float(weight) != 1.0 else ""
        lines.append(f"  [{c['name']}] {weight_suffix}  {c['description']}")
        lines.append(f"    1.0 → {c['score_1']}")
        lines.append(f"    0.5 → {c['score_half']}")
        lines.append(f"    0.0 → {c['score_0']}")
    return "\n".join(lines)


def _criterion_weights(rubric: dict) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for crit in rubric.get("rubric_criteria") or []:
        name = str(crit.get("name") or "").strip()
        if not name:
            continue
        try:
            weight = float(crit.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weights[name] = max(weight, 0.0)
    return weights


def _format_evidence(evidence_texts: List[str]) -> str:
    if not evidence_texts:
        return "(no evidence retrieved yet)"
    return "\n".join(f"[E{i+1}] {t}" for i, t in enumerate(evidence_texts))


def _format_missing_dict(d: dict) -> str:
    """Render structured missing_evidence_analysis as one readable sentence.

    Fields: focus_options, target_fact, time_scope, conflict.
    Used downstream by planner — keeps backward-compat with string consumers.
    """
    parts: List[str] = []
    target = str(d.get("target_fact") or "").strip()
    if target:
        parts.append(target)
    focus = d.get("focus_options") or []
    if isinstance(focus, list) and focus:
        letters = [str(x).strip().upper()[:1] for x in focus if str(x).strip()]
        if letters:
            parts.append(f"(focus options: {', '.join(letters)})")
    scope = str(d.get("time_scope") or "").strip()
    if scope and scope.lower() != "null":
        parts.append(f"around {scope}")
    conflict = str(d.get("conflict") or "").strip()
    if conflict and conflict.lower() != "null":
        parts.append(f"unresolved conflict: {conflict}")
    return " ".join(parts).strip()


# ── Verifier class ─────────────────────────────────────────────────────────────

class Verifier:
    def __init__(self, llm):
        self.llm = llm

    def get_rubric(self, question: str, task_type: Optional[str] = None) -> str:
        return get_rubric(question, task_type)

    def verify(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        rubric: str | dict,
        keyframe_images=(),
        rubric_judgment: bool = True,
        explicit_attribution: bool = False,
        margin_threshold: float = 0.20,
        loose: bool = False,
    ) -> Dict:
        """Judge evidence sufficiency with rubric-guided per-option scoring.

        rubric_judgment=True (default):
            LLM scores each rubric criterion per option; Python computes
            option_scores, option_status, label, and weak_rubric_criteria.
            label: "FULLY_SUFFICIENT" | "ANSWER_SUFFICIENT" | "INSUFFICIENT"

        rubric_judgment=False:
            Legacy no-rubric path; label: "sufficient" | "insufficient"
        """
        if isinstance(rubric, str):
            rubric_dict = {"_legacy_text": rubric, "scoring_rule": "average",
                           "sufficient_threshold": 0.5}
        else:
            rubric_dict = rubric

        # ── No-rubric path ────────────────────────────────────────────────────
        if not rubric_judgment:
            opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
            ev   = _format_evidence(evidence_texts)
            user = "\n\n".join([
                f"Question: {question}", f"Options:\n{opts}",
                f"Evidence Chain:\n{ev}", "Return the JSON now.",
            ])
            messages = [
                {"role": "system", "content": VERIFIER_SYS_NORUBRIC},
                {"role": "user",   "content": user},
            ]
            if keyframe_images and getattr(self.llm, '_api_endpoints', None):
                messages[-1] = _inject_images(messages[-1], keyframe_images)
            raw    = self.llm.chat(messages, max_new_tokens=256, enable_thinking=False)
            parsed = extract_json(raw)
            label  = as_str(parsed.get("label", "insufficient")).lower()
            if label not in ("sufficient", "insufficient"):
                label = "insufficient"
            raw_missing = parsed.get("missing_evidence_analysis") or parsed.get("missing_evidence")
            if isinstance(raw_missing, dict):
                missing = _format_missing_dict(raw_missing) or None
            else:
                missing = raw_missing if raw_missing else None
            raw_opts = parsed.get("option_status") or {}
            option_status: Dict[str, str] = {}
            if isinstance(raw_opts, dict):
                for k, v in raw_opts.items():
                    key = as_str(k).strip().upper()[:1]
                    val = as_str(v).strip().lower()
                    if key and val in ("verified", "excluded", "unclear", "conflicting"):
                        option_status[key] = val
            return {
                "label": label, "option_status": option_status,
                "option_scores": {}, "option_criteria_scores": {},
                "weak_rubric_criteria": [],
                "missing_evidence_analysis": missing if label == "insufficient" else None,
                "score": 0.0, "criteria": {}, "reasoning": as_str(parsed.get("reasoning", "")),
                "evidence_attribution": {}, "key_ids": [], "distractor_ids": [],
            }

        # ── Rubric path ────────────────────────────────────────────────────────
        rubric_section = (
            rubric_dict["_legacy_text"]
            if "_legacy_text" in rubric_dict
            else _format_rubric_for_user(rubric_dict)
        )
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        ev   = _format_evidence(evidence_texts)
        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}",
            "Return the JSON now.",
        ])
        sys_for_rubric = VERIFIER_SYS_LOOSE if loose else VERIFIER_SYS
        messages = [
            {"role": "system", "content": sys_for_rubric},
            {"role": "user",   "content": user},
        ]
        if keyframe_images and getattr(self.llm, '_api_endpoints', None):
            messages[-1] = _inject_images(messages[-1], keyframe_images)
        raw    = self.llm.chat(messages, max_new_tokens=800, enable_thinking=False)
        parsed = extract_json(raw)

        # ── Parse option_criteria_scores ───────────────────────────────────────
        raw_ocs = parsed.get("option_criteria_scores") or {}
        option_criteria_scores: Dict[str, Dict[str, float]] = {}
        if isinstance(raw_ocs, dict):
            for opt_key, crit_dict in raw_ocs.items():
                opt = as_str(opt_key).strip().upper()[:1]
                if not opt or not isinstance(crit_dict, dict):
                    continue
                scores: Dict[str, float] = {}
                for cname, cscore in crit_dict.items():
                    try:
                        scores[str(cname).strip()] = float(cscore)
                    except (TypeError, ValueError):
                        scores[str(cname).strip()] = 0.0
                if scores:
                    option_criteria_scores[opt] = scores

        # ── Compute option_rubric_scores (weighted average per option) ─────────
        weights = _criterion_weights(rubric_dict)
        option_rubric_scores: Dict[str, float] = {}
        for opt, crit_scores in option_criteria_scores.items():
            total_w = weighted_sum = 0.0
            for name, val in crit_scores.items():
                w = weights.get(name, 1.0)
                total_w += w
                weighted_sum += w * val
            option_rubric_scores[opt] = (weighted_sum / total_w) if total_w > 0 else 0.0

        # ── Parse LLM option_judgment, then enforce threshold ──────────────────
        # rubric_score measures evidence sufficiency to judge an option (not support direction).
        # Python overrides to "unknown" when rubric_score < sufficient_threshold.
        sufficient_threshold = rubric_dict.get("sufficient_threshold", 0.75)
        raw_judgment = parsed.get("option_judgment") or {}
        llm_judgment: Dict[str, str] = {}
        if isinstance(raw_judgment, dict):
            for k, v in raw_judgment.items():
                key = as_str(k).strip().upper()[:1]
                val = as_str(v).strip().lower()
                if key and val in ("true", "false", "unknown"):
                    llm_judgment[key] = val

        option_judgment: Dict[str, str] = {}
        all_opts = [chr(ord('A') + i) for i in range(len(candidates))]
        for opt in all_opts:
            rscore = option_rubric_scores.get(opt)
            if rscore is None or rscore < sufficient_threshold:
                option_judgment[opt] = "unknown"
            else:
                option_judgment[opt] = llm_judgment.get(opt, "unknown")

        # ── Compute label ──────────────────────────────────────────────────────
        n_opts      = len(candidates)
        true_opts   = [k for k, v in option_judgment.items() if v == "true"]
        false_opts  = [k for k, v in option_judgment.items() if v == "false"]
        unknown_opts = [k for k, v in option_judgment.items() if v == "unknown"]

        if len(true_opts) == 1 and len(false_opts) == n_opts - 1:
            label = "FULLY_SUFFICIENT"
        elif len(true_opts) == 1 and unknown_opts:
            true_score       = option_rubric_scores[true_opts[0]]
            max_unknown_score = max(option_rubric_scores.get(u, 0.0) for u in unknown_opts)
            margin = true_score - max_unknown_score
            label = "ANSWER_SUFFICIENT" if margin >= margin_threshold else "INSUFFICIENT"
        else:
            label = "INSUFFICIENT"

        # ── Weak rubric criteria (from unknown options only) ───────────────────
        weak_rubric_criteria: List[str] = []
        missing: Optional[str] = None
        if label == "INSUFFICIENT" and unknown_opts:
            crit_totals: Dict[str, float] = {}
            crit_counts: Dict[str, int]   = {}
            for opt in unknown_opts:
                for crit, sc in option_criteria_scores.get(opt, {}).items():
                    crit_totals[crit] = crit_totals.get(crit, 0.0) + sc
                    crit_counts[crit] = crit_counts.get(crit, 0) + 1
            if crit_totals:
                crit_avgs = {k: crit_totals[k] / crit_counts[k] for k in crit_totals}
                weak_rubric_criteria = [
                    k for k, v in sorted(crit_avgs.items(), key=lambda x: x[1])
                    if v < 0.5
                ]
            raw_missing = parsed.get("missing_evidence_analysis")
            if isinstance(raw_missing, dict):
                missing = _format_missing_dict(raw_missing) or missing
            elif isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "option_judgment":        option_judgment,
            "unknown_options":        unknown_opts,
            "option_rubric_scores":   {k: round(v, 4) for k, v in option_rubric_scores.items()},
            "option_criteria_scores": option_criteria_scores,
            "weak_rubric_criteria":   weak_rubric_criteria,
            "missing_evidence_analysis": missing,
            # Legacy fields
            "score":               round(max(option_rubric_scores.values()), 4) if option_rubric_scores else 0.0,
            "criteria":            {},
            "reasoning":           "",
            "evidence_attribution": {},
            "key_ids":             [],
            "distractor_ids":      [],
        }
