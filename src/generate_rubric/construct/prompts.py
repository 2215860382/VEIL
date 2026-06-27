"""Prompts for pairwise evidence-sufficiency rubric construction."""
from __future__ import annotations

import json

from .schema import INITIAL_RUBRIC


PAIRWISE_SYSTEM = """\
You construct evidence-sufficiency rubrics for long-video multiple-choice QA.

Important definitions:
- rubric_score = whether the current evidence chain is sufficient to verify or exclude EACH option.
- rubric_score is NOT answer correctness.
- rubric_score is NOT option support.
- A false option can receive a high sufficiency score when the evidence is enough to exclude it.

Method:
- Follow OnlineRubrics: start from existing rubric criteria, compare two evidence chains, and extract only differences not already covered.
- Follow Chasing the Tail: prioritize differences between high-quality chains; great-vs-excellent differences matter more than bad-vs-good.
- Every new criterion must be grounded in evidence differences actually present in the pair.
- Do not invent criteria from outside knowledge.
- Do not produce generic criteria such as "evidence is relevant/complete/clear/supports the answer".

Output one strict JSON object.
"""


PAIRWISE_USER_TEMPLATE = """\
Existing rubric:
{initial_rubric}

Question type: {question_type}
Question: {question}
Options:
{options}

Weaker chain ({weaker_quality}, id={weaker_id}):
{weaker_evidence}

Stronger chain ({stronger_quality}, id={stronger_id}):
{stronger_evidence}

Analyze the pair in this exact order:
1. What can the weaker chain already verify or exclude?
2. What can the weaker chain NOT verify or exclude?
3. What key evidence does the stronger chain add?
4. Why does the added evidence make specific options verifiable or excludable?
5. Which differences are already covered by the existing rubric?
6. Which differences are NOT covered and should become evidence-sufficiency criteria?

Return JSON:
{{
  "pair_type": "{pair_type}",
  "difference_analysis": {{
    "weaker_can_verify_or_exclude": ["..."],
    "weaker_missing": ["..."],
    "stronger_adds": ["..."],
    "option_level_effect": ["..."],
    "covered_by_existing_rubric": ["..."]
  }},
  "new_criteria": [
    {{
      "name": "short_snake_case",
      "description": "Specific evidence-sufficiency criterion, not answer correctness",
      "question_types": ["{question_type}"],
      "evidence_need": "Concrete fact/time/action/entity/relation evidence required",
      "score_1": "Evidence fully sufficient condition",
      "score_half": "Partially sufficient condition",
      "score_0": "Insufficient condition",
      "repair_action": "refine_query|time_sorted|dense_sample|broadcast|asr_match",
      "source_observation": "Which pair difference produced this criterion"
    }}
  ],
  "discarded_generic_ideas": ["..."]
}}
"""


AGGREGATE_SYSTEM = """\
You aggregate evidence-sufficiency rubric criteria for long-video multiple-choice QA.

Keep only criteria that:
- explain multiple failure cases,
- distinguish weak/good/great/excellent evidence chains,
- help verifier output true/false/unknown for each option,
- guide planner repair queries.

Remove criteria that are generic, duplicate, single-case only, impossible to verify from video/caption/ASR/time, or confuse correctness with sufficiency.

Output strict YAML only.
"""


AGGREGATE_CHUNK_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Candidate criteria extracted from pairwise comparisons:
{criteria_json}

Produce a compact intermediate rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
    source_pair_examples: [pair_id, ...]

Rules:
- Keep at most {max_criteria} criteria.
- Preserve only criteria that are specific and useful for the given question type.
- If multiple candidates overlap, merge them.
- Do not include generic sufficiency language.
"""


AGGREGATE_MERGE_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Intermediate chunk summaries:
{chunk_summaries}

Merge the chunk summaries into one compact type-level rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
    source_pair_examples: [pair_id, ...]

Rules:
- Keep at most {max_criteria} criteria.
- Prefer criteria that distinguish weak/good/great/excellent evidence chains.
- Remove duplicates and generic language.
- Output strict YAML only.
"""


AGGREGATE_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Candidate criteria extracted from pairwise comparisons:
{criteria_json}

Produce a compact type-level rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
scoring_rule: average
sufficient_threshold: 0.80

Rules:
- Keep at most {max_criteria} criteria.
- Include source_pair_examples under each criterion as a YAML list of pair_id values.
- Descriptions must evaluate evidence sufficiency for verifying/excluding options, not answer quality.
"""


def format_initial_rubric() -> str:
    return json.dumps(INITIAL_RUBRIC, ensure_ascii=False, indent=2)


def format_options(candidates: list[str]) -> str:
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))


def format_evidence(items: list[dict], max_chars: int = 9000) -> str:
    parts = []
    for i, ev in enumerate(items, 1):
        ts = ""
        if ev.get("start_time") is not None and ev.get("end_time") is not None:
            ts = f" {ev['start_time']:.1f}-{ev['end_time']:.1f}s"
        parts.append(f"[E{i} cid={ev.get('chunk_id')}{ts}] {ev.get('text', '')}")
    text = "\n".join(parts)
    return text[:max_chars]
