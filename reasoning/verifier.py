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

from utils.jsonx import as_str, extract_json


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
def _rubric_config() -> Tuple[Dict, Dict[str, str], List[Tuple[List[str], str]]]:
    path = Path(__file__).with_name("rubric_templates.yaml")
    if not path.is_file():
        raise FileNotFoundError(
            f"VEIL rubric templates missing: {path}"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    templates: Dict = {}
    for k, v in (data.get("templates") or {}).items():
        templates[k] = v  # dict or str (str = legacy flat template)

    aliases: Dict[str, str] = dict(data.get("type_aliases") or {})
    rules_raw = data.get("keyword_rules") or []
    keyword_rules: List[Tuple[List[str], str]] = []
    for row in rules_raw:
        kws = row.get("keywords") or []
        qtype = row.get("qtype", "default")
        if isinstance(kws, str):
            kws = [kws]
        keyword_rules.append((list(kws), str(qtype)))

    if "default" not in templates:
        raise ValueError("rubric_templates.yaml must define templates.default")
    return templates, aliases, keyword_rules


def get_rubric_dict(question: str, task_type: Optional[str] = None) -> dict:
    """Return the structured rubric dict for the given question / task type.

    The dict has keys: rubric_criteria, scoring_rule, sufficient_threshold
    (all from the YAML template).
    """
    templates, aliases, keyword_rules = _rubric_config()

    key = "default"
    if task_type:
        key = aliases.get(task_type, key)
    if key not in templates:
        key = "default"

    # keyword_rules override (question-text based routing)
    q_lower = (question or "").lower()
    for kws, qtype in keyword_rules:
        if any(kw in q_lower for kw in kws) and qtype in templates:
            key = qtype
            break

    tpl = templates[key]
    if isinstance(tpl, str):
        # Legacy flat-string template: wrap in a minimal dict so callers don't break.
        return {"_legacy_text": tpl, "scoring_rule": "average", "sufficient_threshold": 0.5}
    return tpl


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
You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.
Follow these FIVE steps strictly, then output ONE strict JSON object — no prose, no markdown fences.

## Step 1 — Evidence Attribution (internal)
For each evidence chunk [E1]...[En], judge its role for EACH answer option:
  support  : the evidence directly supports that option
  refute   : the evidence directly contradicts or rules out that option
  neutral  : the evidence has no clear effect on that option
  conflict : the evidence conflicts with other evidence on a key fact for that option
Use this step internally to reason about the evidence.
Do NOT output per-evidence attribution.

## Step 2 — Per-Option Status
Using the internal evidence attribution, judge EACH answer option:
  verified    : evidence clearly supports this option
  excluded    : evidence clearly rules out this option
  unclear     : evidence is insufficient to decide
  conflicting : evidence for this option contains unresolved conflict
Output "option_status" as an object mapping each option letter to one of these labels.
Important:
  - option_status is explicit per-option evidence judgment.
  - conflicting means evidence conflict, NOT that two answer options are mutually exclusive.
  - If evidence clearly supports one option and the options are mutually exclusive, you may mark
    other options excluded, but only if there is no unresolved conflict.

## Step 3 — Rubric Criteria Scoring
For each criterion in the Rubric assign a score:
  1.0 = fully satisfied   0.5 = partially satisfied   0.0 = not satisfied
Rubric scoring may use the internal evidence attribution and option_status. For criteria such as
option_disambiguation, use option_status as supporting judgment.

## Step 4 — Aggregate Score & Label
Compute "score" using the scoring rule stated in the Rubric (average or min).
Do NOT let option_status directly participate in score aggregation; only criterion scores aggregate.
Set "label":
  • "sufficient"   if score ≥ threshold stated in the Rubric
  • "insufficient" otherwise

## Step 5 — Reasoning & Missing Evidence
Write 1-2 sentences in "reasoning" explaining your decision.
If insufficient, output a SEMI-STRUCTURED "missing_evidence_analysis" object based on low-scoring
rubric criteria, unclear options, and conflicting options.
This object will be used directly as guidance for the next retrieval query.
Use these fields:
  - "focus_options": list of option letters still needing evidence (e.g. ["A","C"]); [] if the gap is global
  - "analysis": one dense, actionable sentence or short paragraph describing what evidence is still needed next
  - "time_scope": a concrete time range if needed (e.g. "first 60 seconds of the video"), otherwise null
  - "conflict_fact": the exact conflicting fact to resolve if there is an unresolved contradiction, otherwise null
Make "analysis" specific:
  - Preserve cross-option relations when multiple options remain entangled
  - For temporal gaps: name the events whose order must be established
  - For fact verification: state the exact claim that must be checked
  - If multiple facts are missing, keep them in one coherent repair directive rather than splitting into isolated bullet-like goals

Return ONLY this JSON (fill every field; use actual criterion names from the Rubric):
{
  "option_status":    {"A": "verified", "B": "excluded"},
  "criteria":         {"<criterion_name>": 0.0},
  "score":            0.0,
  "label":            "sufficient" or "insufficient",
  "reasoning":        "...",
  "missing_evidence_analysis": {
    "focus_options": ["A"],
    "analysis": "Need direct evidence clarifying whether Whitehead's very first flight attempt succeeded or failed, and whether later successful flights are being conflated with the first attempt.",
    "time_scope": null,
    "conflict_fact": null
  }
}"""


VERIFIER_SYS_WITH_ATTR = """\
You judge whether retrieved video-segment evidence is sufficient to answer a multiple-choice question.
Follow these FIVE steps strictly, then output ONE strict JSON object — no prose, no markdown fences.

## Step 1 — Evidence Attribution
For each evidence chunk [E1]...[En], judge its role for EACH answer option:
  support  : the evidence directly supports that option
  refute   : the evidence directly contradicts or rules out that option
  neutral  : the evidence has no clear effect on that option
  conflict : the evidence conflicts with other evidence on a key fact for that option
Output "evidence_attribution" as an object keyed by evidence id, then option letter.
Also output compatibility summaries:
  "key_ids"        : evidence ids that directly help identify the answer or exclude wrong options
  "distractor_ids" : evidence ids that are misleading, contradictory, or conflict-heavy

## Step 2 — Per-Option Status
Using Evidence Attribution, judge EACH answer option:
  verified    : evidence clearly supports this option
  excluded    : evidence clearly rules out this option
  unclear     : evidence is insufficient to decide
  conflicting : evidence for this option contains unresolved conflict
Output "option_status" as an object mapping each option letter to one of these labels.
Important:
  - option_status is explicit per-option evidence judgment.
  - conflicting means evidence conflict, NOT that two answer options are mutually exclusive.
  - If evidence clearly supports one option and the options are mutually exclusive, you may mark
    other options excluded, but only if there is no unresolved conflict.

## Step 3 — Rubric Criteria Scoring
For each criterion in the Rubric assign a score:
  1.0 = fully satisfied   0.5 = partially satisfied   0.0 = not satisfied
Rubric scoring may use Evidence Attribution and option_status. For criteria such as
option_disambiguation, use option_status as supporting judgment.

## Step 4 — Aggregate Score & Label
Compute "score" using the scoring rule stated in the Rubric (average or min).
Do NOT let option_status directly participate in score aggregation; only criterion scores aggregate.
Set "label":
  • "sufficient"   if score ≥ threshold stated in the Rubric
  • "insufficient" otherwise

## Step 5 — Reasoning & Missing Evidence
Write 1-2 sentences in "reasoning" explaining your decision.
If insufficient, output a SEMI-STRUCTURED "missing_evidence_analysis" object based on low-scoring
rubric criteria, unclear options, and conflicting options.
This object will be used directly as guidance for the next retrieval query.
Use these fields:
  - "focus_options": list of option letters still needing evidence (e.g. ["A","C"]); [] if the gap is global
  - "analysis": one dense, actionable sentence or short paragraph describing what evidence is still needed next
  - "time_scope": a concrete time range if needed (e.g. "first 60 seconds of the video"), otherwise null
  - "conflict_fact": the exact conflicting fact to resolve if there is an unresolved contradiction, otherwise null

Return ONLY this JSON (fill every field; use actual criterion names from the Rubric):
{
  "evidence_attribution": {"E1": {"A": "support", "B": "refute"}},
  "option_status":    {"A": "verified", "B": "excluded"},
  "criteria":         {"<criterion_name>": 0.0},
  "score":            0.0,
  "label":            "sufficient" or "insufficient",
  "reasoning":        "...",
  "missing_evidence_analysis": {
    "focus_options": ["A"],
    "analysis": "Need direct evidence clarifying whether Whitehead's very first flight attempt succeeded or failed, and whether later successful flights are being conflated with the first attempt.",
    "time_scope": null,
    "conflict_fact": null
  },
  "key_ids":          [1],
  "distractor_ids":   []
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
        use_rubric_judgment: bool = True,
        include_evidence_attribution: bool = False,
    ) -> Dict:
        """Judge evidence sufficiency with rubric-guided structured reasoning.

        Args:
            rubric: Either the structured dict from get_rubric_dict() (preferred)
                    or a legacy text string (backward-compat).

        Returns dict with keys:
            label           – "sufficient" | "insufficient"
            missing_evidence_analysis – None or semi-structured repair guidance
            score           – float aggregate of rubric criteria
            criteria        – {criterion_name: score} (empty for legacy rubric)
            evidence_attribution – optional per-evidence, per-option support/refute/neutral/conflict
            option_status   – per-option verified/excluded/unclear/conflicting judgment
            reasoning       – brief explanation string
        """
        # Accept both dict and string rubric.
        if isinstance(rubric, str):
            rubric_dict = {"_legacy_text": rubric, "scoring_rule": "average",
                           "sufficient_threshold": 0.5}
        else:
            rubric_dict = rubric

        if not use_rubric_judgment:
            # No rubric: LLM makes holistic sufficient/insufficient judgment
            sys_prompt = VERIFIER_SYS_NORUBRIC
            rubric_section = ""
        else:
            sys_prompt = VERIFIER_SYS_WITH_ATTR if include_evidence_attribution else VERIFIER_SYS
            rubric_section = (
                rubric_dict["_legacy_text"]
                if "_legacy_text" in rubric_dict
                else _format_rubric_for_user(rubric_dict)
            )

        opts = "\n".join(f"  ({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))
        ev   = _format_evidence(evidence_texts)

        user_parts = [
            f"Question: {question}",
            f"Options:\n{opts}",
        ]
        if rubric_section:
            user_parts.append(f"Rubric:\n{rubric_section}")
        user_parts.append(f"Evidence Chain:\n{ev}")
        user_parts.append("Return the JSON now.")
        user = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user},
        ]
        if keyframe_images and getattr(self.llm, '_api_endpoints', None):
            messages[-1] = _inject_images(messages[-1], keyframe_images)
        if not use_rubric_judgment:
            max_new_tokens = 256
        else:
            max_new_tokens = 640
        raw    = self.llm.chat(messages, max_new_tokens=max_new_tokens, enable_thinking=False)
        parsed = extract_json(raw)

        # ── Extract fields ─────────────────────────────────────────────────────
        label = as_str(parsed.get("label", "insufficient")).lower()
        if label not in ("sufficient", "insufficient"):
            label = "insufficient"

        # Criteria scores
        raw_criteria = parsed.get("criteria")
        criteria_scores: Dict[str, float] = {}
        if isinstance(raw_criteria, dict):
            for k, v in raw_criteria.items():
                try:
                    criteria_scores[k] = float(v)
                except (TypeError, ValueError):
                    criteria_scores[k] = 0.0

        # Aggregate score
        try:
            raw_score = float(parsed.get("score", 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0

        if criteria_scores:
            rule = rubric_dict.get("scoring_rule", "average")
            if rule == "min":
                agg_score = min(criteria_scores.values())
            else:
                weights = _criterion_weights(rubric_dict)
                total_weight = 0.0
                weighted_sum = 0.0
                for name, value in criteria_scores.items():
                    w = weights.get(name, 1.0)
                    total_weight += w
                    weighted_sum += w * value
                agg_score = (weighted_sum / total_weight) if total_weight > 0 else statistics.mean(criteria_scores.values())
            score = agg_score
        else:
            score = raw_score

        reasoning = as_str(parsed.get("reasoning", ""))

        # missing_evidence_analysis — semi-structured object preferred; string kept for compatibility
        raw_missing = parsed.get("missing_evidence_analysis")
        if raw_missing is None:
            # Backward compatibility with older verifier outputs.
            raw_missing = parsed.get("missing_evidence")
        if label == "sufficient" or not raw_missing:
            missing = None
        elif isinstance(raw_missing, dict):
            focus_options = raw_missing.get("focus_options")
            if not isinstance(focus_options, list):
                focus_options = []
            focus_options = [
                as_str(x).strip().upper()[:1]
                for x in focus_options
                if as_str(x).strip()
            ]

            analysis = as_str(raw_missing.get("analysis", "")).strip()
            time_scope = as_str(raw_missing.get("time_scope", "")).strip() or None
            conflict_fact = as_str(raw_missing.get("conflict_fact", "")).strip() or None

            # Backward compatibility for older dict outputs.
            if not analysis:
                desc = as_str(raw_missing.get("description", "")).strip()
                if desc:
                    analysis = desc
            if not analysis:
                query_goals = raw_missing.get("query_goals")
                if not isinstance(query_goals, list):
                    query_goals = []
                query_goals = [as_str(x).strip() for x in query_goals if as_str(x).strip()]
                if query_goals:
                    analysis = " ; ".join(query_goals)

            missing = {
                "focus_options": focus_options,
                "analysis": analysis or None,
                "time_scope": time_scope,
                "conflict_fact": conflict_fact,
            }
        elif isinstance(raw_missing, str):
            missing = raw_missing or None
        else:
            missing = None

        # evidence_attribution: per-evidence, per-option support/refute/neutral/conflict
        raw_attr = parsed.get("evidence_attribution") or {}
        evidence_attribution: Dict[str, Dict[str, str]] = {}
        if isinstance(raw_attr, dict):
            for ev_key, option_map in raw_attr.items():
                ev = as_str(ev_key).strip()
                if not ev or not isinstance(option_map, dict):
                    continue
                norm_map: Dict[str, str] = {}
                for opt_key, role in option_map.items():
                    key = as_str(opt_key).strip().upper()[:1]
                    val = as_str(role).strip().lower()
                    if key and val in ("support", "refute", "neutral", "conflict"):
                        norm_map[key] = val
                if norm_map:
                    evidence_attribution[ev] = norm_map

        # key_ids / distractor_ids (compatibility summaries for evidence attribution)
        def _parse_id_list(raw) -> List[int]:
            if not isinstance(raw, list):
                return []
            return [int(x) for x in raw if isinstance(x, (int, float)) and int(x) >= 1]

        key_ids        = _parse_id_list(parsed.get("key_ids"))
        distractor_ids = _parse_id_list(parsed.get("distractor_ids"))
        if include_evidence_attribution and use_rubric_judgment and label == "sufficient" and not key_ids:
            # fallback: treat all evidence as key when sufficient and no attribution given
            key_ids = list(range(1, len(evidence_texts) + 1))

        # option_status: per-option verified/excluded/unclear/conflicting judgment
        raw_opts = parsed.get("option_status") or {}
        option_status: Dict[str, str] = {}
        if isinstance(raw_opts, dict):
            for k, v in raw_opts.items():
                key = as_str(k).strip().upper()[:1]
                val = as_str(v).strip().lower()
                if key and val in ("verified", "excluded", "unclear", "conflicting"):
                    option_status[key] = val

        return {
            "label":            label,
            "missing_evidence_analysis": missing,
            "score":           round(score, 4),
            "criteria":        criteria_scores,
            "reasoning":       reasoning,
            "evidence_attribution": evidence_attribution,
            "key_ids":         key_ids,
            "distractor_ids":  distractor_ids,
            "option_status":   option_status,
        }
