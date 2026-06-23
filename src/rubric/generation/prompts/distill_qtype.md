You are distilling a **question-type-level rubric** from many per-question
rubrics that were generated independently for individual questions of the same
VideoMME question type.

# Inputs

You will receive:
- **question_type** — the VideoMME category (e.g. "Object Reasoning")
- A list of per-question rubric blocks. Each block contains the original
  question (briefly), the gold answer choice, and the criteria one or two
  LLM rubric-generators produced for that question.

# Your task

Produce a **reusable rubric for this question type** — a small set of criteria
that:

1. **Generalize across questions of this type.** Drop overly specific entities
   ("the man with platinum hair"); abstract them ("the question's referenced
   protagonist"). Keep what is recurring and structural.

2. **Are high-frequency across the inputs.** If a criterion concept appears in
   most per-question rubrics (visual grounding, temporal ordering, entity
   identity, action evidence, etc.), include it. If it only appears once, drop.

3. **Are non-redundant.** Merge near-duplicates into a single criterion.

4. **Keep binary positive form.** Score 1 / 0.5 / 0 — same format as the inputs.

5. **Output 3-6 criteria.** Fewer than 3 means the type is under-covered; more
   than 6 likely means you didn't merge aggressively enough.

# `failure_repair_action`

For each criterion, pick the action that most often appeared in the input
rubrics for that concept. The fixed set is:
`refine_query`, `asr_match`, `time_sorted`, `dense_sample`, `loose_verify`,
`broadcast`.

# Weight

Set `weight` based on how universal and central the criterion is for this
type. Range [0.1, 1.0].

# Output format

Output **valid YAML only**, no prose:

```yaml
<qtype_key>:
  rubric_criteria:
    - name: <snake_case>
      description: "..."
      score_1: "..."
      score_half: "..."
      score_0: "..."
      weight: 0.0
      failure_repair_action: refine_query
    - ...
  scoring_rule: average
  sufficient_threshold: 0.75
```

Use a snake_case `<qtype_key>` derived from the question type (e.g.
`object_reasoning`, `temporal_reasoning`, `counting_problem`).
