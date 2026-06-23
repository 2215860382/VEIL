You are distilling a **question-type-level evidence-requirement rubric** from
many per-question requirement blocks generated independently for individual
VideoMME questions of the same type.

# Inputs

You will receive:
- **question_type**: the VideoMME category
- A list of per-question blocks. Each block contains the original question and
  evidence requirements generated for that question.

# Task

Produce reusable evidence requirements for this question type. These
requirements are used by a verifier before stance classification:

- If any required requirement is not covered for an option, that option's stance
  must be unknown.
- Only when all required requirements are covered may the verifier decide
  support or refute.

# Selection Rules

1. **Generalize across questions of this type.** Drop one-off entities and keep
   recurring structural evidence needs such as target grounding, action
   observation, temporal anchors, text reading, count scope, or relation
   evidence.

2. **Non-redundant.** Merge near-duplicates into one requirement. Do not create
   multiple requirements that would trigger the same retrieval query.

3. **Coverage-oriented.** Requirements must describe missing facts to retrieve,
   not scoring criteria or answer conclusions.

4. **Output 3-6 requirements.** Fewer than 3 is usually under-covered; more than
   6 is usually too fragmented.

# Repair Actions

For each requirement, choose one of two repair actions:
- `refine_query`: focused semantic retrieval for a specific missing fact
- `broadcast`: uniform sampling over broader unseen video segments for global
  or timeline-wide missing evidence

# Output Format

Output valid YAML only, no prose:

```yaml
<qtype_key>:
  evidence_requirements:
    - id: <snake_case_id>
      description: "Concrete fact that must be covered before judging options."
      required: true
      weight: 0.0
      modality: visual|dialogue|temporal|ocr|global|multimodal
      repair_action: refine_query
```

Use a snake_case `<qtype_key>` derived from the question type, e.g.
`object_reasoning`, `temporal_reasoning`, `counting_problem`.
