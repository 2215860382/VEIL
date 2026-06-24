"""Verifier — judge whether retrieved evidence satisfies the rubric for this question type.

Rubric-guided judgment:
  1. Evidence attribution – per-evidence, per-option support / refute / neutral / conflict.
  2. Option status        – verified / excluded / unclear / conflicting for each option.
  3. Rubric criteria      – explicit 0 / 0.5 / 1 per criterion; aggregated to a score.
  4. Label and gaps       – label follows the rubric threshold; gaps drive the next query.

Rubric lives in ``src/rubric/rubric_templates.yaml`` (structured YAML, pluggable per task_type).
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
    path = Path(__file__).resolve().parents[1] / "rubric" / "rubric_templates.yaml"
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

## Step 1 — Per-Option Evidence Sufficiency Scoring
For each answer option, convert it into a proposition and score each rubric criterion:
  1.0 = evidence fully sufficient to judge this option on this criterion
        (clearly confirms OR clearly rules it out — both count as sufficient)
  0.5 = evidence partially sufficient — some relevant information present but incomplete
  0.0 = evidence insufficient to judge this option on this criterion

Score whether the evidence ALLOWS JUDGMENT of this option, not whether it supports it.
Evidence that clearly rules out an option scores as high as evidence that clearly confirms it.
When unsure between 1.0 and 0.5, choose 0.5; between 0.5 and 0.0, choose 0.0.

## Step 2 — Option Judgment
Based on the overall evidence for each option:
  true    : evidence clearly establishes this option is correct
  false   : evidence clearly establishes this option is incorrect
  unknown : evidence is insufficient to determine whether this option is correct or not

## Step 3 — Missing Evidence Analysis
If any option is "unknown", write ONE actionable sentence in "missing_evidence_analysis"
naming the specific fact, time range, or event still needed to resolve it.
Leave it empty string if no option is unknown.

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "option_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  },
  "option_judgment": {
    "A": "true|false|unknown",
    "B": "true|false|unknown"
  },
  "missing_evidence_analysis": "..."
}"""

VERIFIER_SYS_WITH_ATTR = VERIFIER_SYS

# Looser variant — explicitly allows synthesizing indirect / partial evidence
# instead of requiring verbatim restatement of the option.
VERIFIER_SYS_LOOSE = """\
You evaluate whether retrieved video evidence is sufficient to judge each answer option.
Output ONE strict JSON object — no prose, no markdown fences.

## CRITICAL — How to read evidence
Video memory is a high-level summary; it almost never restates an answer option verbatim.
Treat evidence as INDIRECT and use these rules:
  * If multiple evidence pieces together IMPLY an option (e.g. mentioning prices,
    discussing $10 / $4000 → "price is a factor"), treat that as sufficient to JUDGE.
  * If the speaker quotes a motive in dialogue (Speech / asr), that quote counts
    as an explicit statement even if paraphrased.
  * Visual descriptions (objects, actions, locations) count as judgable evidence
    for perception / counting / spatial / temporal-order claims.
  * Only return "unknown" when the evidence is genuinely off-topic, NOT just because
    it lacks word-for-word phrasing of the option.

## Step 1 — Per-Option Evidence Sufficiency Scoring
For each option, score each rubric criterion:
  1.0 = the cumulative evidence ALLOWS a judgment (confirm OR rule out)
        — including via reasonable inference from indirect evidence.
  0.5 = some relevant signal but truly ambiguous.
  0.0 = no usable signal at all for this criterion.

## Step 2 — Option Judgment
  true    : evidence (direct or by reasonable inference) establishes the option is correct
  false   : evidence establishes the option is incorrect or contradicts it
  unknown : evidence is genuinely silent on this option

Prefer true/false over unknown whenever the evidence permits a reasonable inference.

## Step 3 — Missing Evidence Analysis
If any option is "unknown", write ONE actionable sentence naming the specific
fact still needed. Leave empty if no option is unknown.

Return ONLY this JSON:
{
  "option_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  },
  "option_judgment": {
    "A": "true|false|unknown",
    "B": "true|false|unknown"
  },
  "missing_evidence_analysis": "..."
}"""


VERIFIER_SYS_NORUBRIC_ATTR = """\
You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.
No rubric scoring. Follow these THREE steps, then output ONE strict JSON object.

## Step 1 — Evidence Attribution (internal)
For each evidence chunk [E1]...[En], judge its role for EACH answer option:
  support  : the evidence directly supports that option
  refute   : the evidence directly contradicts or rules out that option
  neutral  : the evidence has no clear effect on that option
  conflict : the evidence conflicts with other evidence on a key fact for that option
Use this step internally to reason about the evidence. Do NOT output per-evidence attribution.

## Step 2 — Holistic Sufficiency Judgment
Based on your internal evidence attribution, decide whether the evidence is sufficient to confidently pick an answer.
  • "sufficient"   if you are confident the evidence supports a specific option
  • "insufficient" otherwise

## Step 3 — Reasoning & Missing Evidence
Write 1-2 sentences in "reasoning" explaining your decision.
If insufficient, output a SEMI-STRUCTURED "missing_evidence_analysis" object with:
  - "focus_options": []
  - "analysis": one dense, actionable sentence describing what evidence is still needed
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
    "analysis": "...",
    "time_scope": null,
    "conflict_fact": null
  }
}"""


VERIFIER_SYS_NORUBRIC_ATTR_OPSTATUS = """\
You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.
No rubric scoring. Follow these FOUR steps, then output ONE strict JSON object.

## Step 1 — Evidence Attribution (internal)
For each evidence chunk [E1]...[En], judge its role for EACH answer option:
  support  : the evidence directly supports that option
  refute   : the evidence directly contradicts or rules out that option
  neutral  : the evidence has no clear effect on that option
  conflict : the evidence conflicts with other evidence on a key fact for that option
Use this step internally to reason about the evidence. Do NOT output per-evidence attribution.

## Step 2 — Per-Option Status & Distractor Identification
Using the internal evidence attribution, judge EACH answer option:
  verified    : evidence EXPLICITLY states the specific fact that maps to this option
  excluded    : evidence EXPLICITLY contradicts or rules out this option with a concrete fact
  unclear     : evidence is insufficient, ambiguous, or only implied — DEFAULT when in doubt
  conflicting : evidence for this option contains unresolved contradiction
Output "option_status" as an object mapping each option letter to one of these labels.
Also output "distractor_ids": list of evidence chunk numbers (1-indexed) that are misleading,
contradictory, or conflict-heavy across options.
RULES:
  - Use "verified" when the evidence clearly supports an option, including by direct inference or strong implication.
  - Use "excluded" when the evidence contradicts or makes an option implausible, including by mutual exclusivity.
  - Use "unclear" only when the evidence is genuinely ambiguous or absent for that option.

## Step 3 — Holistic Sufficiency Judgment
Based on option_status, decide whether the evidence is sufficient holistically.
Do NOT apply a rubric score threshold — use your own judgment about whether you can confidently answer.
  • "sufficient"   if you can identify a clear answer from the evidence
  • "insufficient" otherwise

## Step 4 — Reasoning & Missing Evidence
Write 1-2 sentences in "reasoning" explaining your decision.
If insufficient, output a SEMI-STRUCTURED "missing_evidence_analysis" object with:
  - "focus_options": list of option letters still unclear or conflicting (e.g. ["A","C"])
  - "analysis": one dense, actionable sentence describing what evidence is still needed
  - "time_scope": concrete time range if needed, otherwise null
  - "conflict_fact": exact conflicting fact if any, otherwise null

Return ONLY this JSON:
{
  "criteria":         {},
  "score":            0.0,
  "option_status":    {"A": "unclear", "B": "unclear"},
  "distractor_ids":   [],
  "reasoning":        "...",
  "label":            "sufficient" or "insufficient",
  "missing_evidence_analysis": {
    "focus_options": ["A"],
    "analysis": "...",
    "time_scope": null,
    "conflict_fact": null
  }
}"""


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
        verifier_attr: bool = False,
        verifier_opstatus: bool = False,
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

        # ── No-rubric path (unchanged) ─────────────────────────────────────────
        if not rubric_judgment:
            if verifier_opstatus:
                sys_prompt = VERIFIER_SYS_NORUBRIC_ATTR_OPSTATUS
            elif verifier_attr:
                sys_prompt = VERIFIER_SYS_NORUBRIC_ATTR
            else:
                sys_prompt = VERIFIER_SYS_NORUBRIC
            opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
            ev   = _format_evidence(evidence_texts)
            user = "\n\n".join([
                f"Question: {question}", f"Options:\n{opts}",
                f"Evidence Chain:\n{ev}", "Return the JSON now.",
            ])
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user},
            ]
            if keyframe_images and getattr(self.llm, '_api_endpoints', None):
                messages[-1] = _inject_images(messages[-1], keyframe_images)
            max_new_tokens = 512 if verifier_opstatus else (384 if verifier_attr else 256)
            raw    = self.llm.chat(messages, max_new_tokens=max_new_tokens, enable_thinking=False)
            parsed = extract_json(raw)
            label  = as_str(parsed.get("label", "insufficient")).lower()
            if label not in ("sufficient", "insufficient"):
                label = "insufficient"
            raw_missing = parsed.get("missing_evidence_analysis") or parsed.get("missing_evidence")
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
            if isinstance(raw_missing, str) and raw_missing.strip():
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
