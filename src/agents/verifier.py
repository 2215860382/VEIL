"""Verifier — judge whether retrieved evidence satisfies the rubric for this question type.

Rubric-guided judgment:
  1. Rubric criteria – explicit 0 / 0.5 / 1 per criterion per option; weighted-averaged
                       to a per-option score, thresholded into true / false / unknown.
  2. Label and gaps  – label follows the rubric threshold; gaps drive the next query.

Default rubric lives in ``outputs/rubric/direct_answer_generated_v2.yaml``.
Currently all questions use ``templates.default``; extend ``type_aliases`` / ``keyword_rules``
in the yaml to route different types to different templates without code changes.
"""
from __future__ import annotations

import statistics
from functools import lru_cache
import os
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
    override = os.environ.get("VEIL_RUBRIC_PATH")
    path = (
        Path(override).expanduser()
        if override
        else Path(__file__).resolve().parents[2]
        / "outputs"
        / "rubric"
        / "direct_answer_generated_v2.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"VEIL rubric templates missing: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    general: Dict = data.get("general") or {}
    if not general.get("rubric_criteria"):
        raise ValueError(f"{path} must define general.rubric_criteria")

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



# ── Two-pass rubric verifier (--verifier-two-pass) ───────────────────────────
# Pass 1 scores sufficiency ONLY (no true/false), so the model is not anchored by
# its own answer guess. Python then thresholds per option, and Pass 2 judges
# true/false conditioned on which options cleared the threshold.
VERIFIER_SYS_TWOPASS_SCORE = """\
You evaluate whether retrieved video evidence is SUFFICIENT to judge each answer option.
Do NOT decide which option is correct — only score evidence sufficiency.
Output ONE strict JSON object — no prose, no markdown fences.

## Per-Option Evidence Sufficiency Scoring
For each answer option, convert it into a proposition and score each rubric criterion:
  1.0 = evidence fully sufficient to judge this option on this criterion
        (clearly confirms OR clearly rules it out — both count as sufficient)
  0.5 = evidence partially sufficient — some relevant information present but incomplete
  0.0 = evidence insufficient to judge this option on this criterion
Score whether the evidence ALLOWS JUDGMENT of this option, not whether it supports it.
Evidence that clearly rules out an option scores as high as evidence that confirms it.
When unsure between 1.0 and 0.5, choose 0.5; between 0.5 and 0.0, choose 0.0.

Return ONLY this JSON (use actual criterion names from the Rubric):
{
  "option_criteria_scores": {
    "A": {"<criterion>": 0.0, ...},
    "B": {"<criterion>": 0.0, ...}
  }
}"""

VERIFIER_SYS_TWOPASS_JUDGE = """\
You judge multiple-choice answer options for a video question. A separate scoring step
already decided which options have SUFFICIENT evidence to judge and which do not; each
option below is tagged [SUFFICIENT] or [INSUFFICIENT].
Output ONE strict JSON object — no prose, no markdown fences.

## Step 1 — Judge the [SUFFICIENT] options
For each option tagged [SUFFICIENT], commit to:
  true   : evidence clearly establishes this option is correct
  false  : evidence clearly establishes this option is incorrect
Do NOT output a judgment for [INSUFFICIENT] options — they are already "unknown".

## Step 2 — Missing Evidence Analysis
If any option is [INSUFFICIENT], write ONE actionable sentence in "missing_evidence_analysis"
naming the specific fact, time range, or event still needed to resolve it.
Leave it empty string if every option is [SUFFICIENT].

Return ONLY this JSON:
{
  "option_judgment": { "A": "true|false", ... },
  "missing_evidence_analysis": "..."
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

    def _parse_criteria_scores(self, parsed: dict) -> Dict[str, Dict[str, float]]:
        """Parse the LLM's option_criteria_scores into {opt: {criterion: float}}."""
        out: Dict[str, Dict[str, float]] = {}
        raw = parsed.get("option_criteria_scores") or {}
        if isinstance(raw, dict):
            for opt_key, crit_dict in raw.items():
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
                    out[opt] = scores
        return out

    def _verify_twopass(
        self, question, candidates, opts, rubric_section, ev, rubric_dict,
        keyframe_images, sufficient_threshold_delta,
    ) -> Dict:
        """Two-pass rubric verify: pass 1 scores sufficiency, pass 2 judges t/f.

        Pass 1 outputs ONLY option_criteria_scores (no answer guess). Python
        thresholds per option, then pass 2 commits true/false on the options that
        cleared the threshold and writes missing_evidence_analysis for the rest.
        Returns the same 6-field verdict as the single-pass rubric path.
        """
        has_imgs = bool(keyframe_images) and bool(getattr(self.llm, '_api_endpoints', None))

        # ── Pass 1: sufficiency scoring only ──────────────────────────────────
        sys1 = os.environ.get("VEIL_VERIFIER_SYS_SCORE", "").strip() or VERIFIER_SYS_TWOPASS_SCORE
        user1 = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}",
            "Return the sufficiency-scores JSON now.",
        ])
        msg1 = [{"role": "system", "content": sys1}, {"role": "user", "content": user1}]
        if has_imgs:
            msg1[-1] = _inject_images(msg1[-1], keyframe_images)
        parsed1 = extract_json(self.llm.chat(msg1, max_new_tokens=600, enable_thinking=False))

        option_criteria_scores = self._parse_criteria_scores(parsed1)

        weights = _criterion_weights(rubric_dict)
        option_rubric_scores: Dict[str, float] = {}
        for opt, crit_scores in option_criteria_scores.items():
            total_w = weighted_sum = 0.0
            for name, val in crit_scores.items():
                w = weights.get(name, 1.0)
                total_w += w
                weighted_sum += w * val
            option_rubric_scores[opt] = (weighted_sum / total_w) if total_w > 0 else 0.0

        sufficient_threshold = min(
            1.0,
            float(rubric_dict.get("sufficient_threshold", 0.75)) + float(sufficient_threshold_delta),
        )
        all_opts   = [chr(ord('A') + i) for i in range(len(candidates))]
        suff_opts  = [o for o in all_opts if option_rubric_scores.get(o, 0.0) >= sufficient_threshold]
        insuff_set = {o for o in all_opts if o not in suff_opts}

        # ── Pass 2: judge the sufficient options, ask missing for the rest ────
        opts_tagged = "\n".join(
            f"  ({chr(ord('A') + i)}) {c} "
            + ("[INSUFFICIENT]" if chr(ord('A') + i) in insuff_set else "[SUFFICIENT]")
            for i, c in enumerate(candidates)
        )
        sys2 = os.environ.get("VEIL_VERIFIER_SYS_JUDGE", "").strip() or VERIFIER_SYS_TWOPASS_JUDGE
        user2 = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts_tagged}",
            f"Evidence Chain:\n{ev}", "Return the judgment JSON now.",
        ])
        msg2 = [{"role": "system", "content": sys2}, {"role": "user", "content": user2}]
        if has_imgs:
            msg2[-1] = _inject_images(msg2[-1], keyframe_images)
        parsed2 = extract_json(self.llm.chat(msg2, max_new_tokens=400, enable_thinking=False))

        raw_judgment = parsed2.get("option_judgment") or {}
        llm_judgment: Dict[str, str] = {}
        if isinstance(raw_judgment, dict):
            for k, v in raw_judgment.items():
                key = as_str(k).strip().upper()[:1]
                val = as_str(v).strip().lower()
                if key and val in ("true", "false", "unknown"):
                    llm_judgment[key] = val

        option_judgment: Dict[str, str] = {}
        for opt in all_opts:
            option_judgment[opt] = "unknown" if opt in insuff_set else llm_judgment.get(opt, "unknown")

        n_opts       = len(candidates)
        true_opts    = [k for k, v in option_judgment.items() if v == "true"]
        false_opts   = [k for k, v in option_judgment.items() if v == "false"]
        unknown_opts = [k for k, v in option_judgment.items() if v == "unknown"]
        label = "SUFFICIENT" if len(true_opts) == 1 and len(false_opts) == n_opts - 1 else "INSUFFICIENT"

        missing: Optional[str] = None
        if label == "INSUFFICIENT" and unknown_opts:
            raw_missing = parsed2.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "option_judgment":        option_judgment,
            "unknown_options":        unknown_opts,
            "option_rubric_scores":   {k: round(v, 4) for k, v in option_rubric_scores.items()},
            "option_criteria_scores": option_criteria_scores,
            "missing_evidence_analysis": missing,
        }

    def verify(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        rubric: str | dict,
        keyframe_images=(),
        sufficient_threshold_delta: float = 0.0,
        two_pass: bool = False,
    ) -> Dict:
        """Judge evidence sufficiency with rubric-guided per-option scoring.

        LLM scores each rubric criterion per option; Python computes the weighted
        per-option score, thresholds it, and derives the label
        ("SUFFICIENT" | "INSUFFICIENT").
        """
        if isinstance(rubric, str):
            rubric_dict = {"_legacy_text": rubric, "scoring_rule": "average",
                           "sufficient_threshold": 0.5}
        else:
            rubric_dict = rubric

        # ── Rubric path ────────────────────────────────────────────────────────
        rubric_section = (
            rubric_dict["_legacy_text"]
            if "_legacy_text" in rubric_dict
            else _format_rubric_for_user(rubric_dict)
        )
        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        ev   = _format_evidence(evidence_texts)

        if two_pass:
            return self._verify_twopass(
                question, candidates, opts, rubric_section, ev, rubric_dict,
                keyframe_images, sufficient_threshold_delta,
            )

        user = "\n\n".join([
            f"Question: {question}", f"Options:\n{opts}",
            f"Rubric:\n{rubric_section}", f"Evidence Chain:\n{ev}",
            "Return the JSON now.",
        ])
        # Prompt-iteration override (no source edit needed); empty/unset → default.
        sys_for_rubric = os.environ.get("VEIL_VERIFIER_SYS", "").strip() or VERIFIER_SYS
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
        sufficient_threshold = min(
            1.0,
            float(rubric_dict.get("sufficient_threshold", 0.75)) + float(sufficient_threshold_delta),
        )
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
                # Threshold only gates whether evidence is sufficient to make a
                # judgment; the LLM may still leave the option unresolved.
                option_judgment[opt] = llm_judgment.get(opt, "unknown")

        # ── Compute label ──────────────────────────────────────────────────────
        n_opts      = len(candidates)
        true_opts   = [k for k, v in option_judgment.items() if v == "true"]
        false_opts  = [k for k, v in option_judgment.items() if v == "false"]
        unknown_opts = [k for k, v in option_judgment.items() if v == "unknown"]

        label = "SUFFICIENT" if len(true_opts) == 1 and len(false_opts) == n_opts - 1 else "INSUFFICIENT"

        # ── Missing evidence analysis (from unknown options only) ──────────────
        missing: Optional[str] = None
        if label == "INSUFFICIENT" and unknown_opts:
            raw_missing = parsed.get("missing_evidence_analysis")
            if isinstance(raw_missing, str) and raw_missing.strip():
                missing = raw_missing.strip()

        return {
            "label":                  label,
            "option_judgment":        option_judgment,
            "unknown_options":        unknown_opts,
            "option_rubric_scores":   {k: round(v, 4) for k, v in option_rubric_scores.items()},
            "option_criteria_scores": option_criteria_scores,
            "missing_evidence_analysis": missing,
        }
