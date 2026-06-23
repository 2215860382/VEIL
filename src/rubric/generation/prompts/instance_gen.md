You are designing a **per-question evaluation rubric** for a video QA verifier.

The verifier sees a candidate answer choice + retrieved video evidence, and must
score whether the evidence is sufficient to support that answer. Your job is to
write the criteria the verifier uses for THIS specific question.

# Inputs

You will receive:
- **question** — the natural-language question
- **candidates** — four answer options A/B/C/D
- **gold** — the correct answer (use as a "reference" to ground rubric criteria,
  *do not* include the literal answer in any rubric criterion)
- **question_type** — VideoMME category (Object Reasoning / Temporal Reasoning / ...)

# Design principles (follow all four)

1. **Expert-grounding.** Anchor each criterion in concrete facts/relations needed
   to deduce the gold answer from video evidence. Do not invent abstract criteria
   disconnected from this specific question.

2. **Comprehensive coverage.** Cover the key dimensions the verifier should
   check: which entities must be located, which actions/states must be observed,
   what temporal/spatial relationships must hold, etc. Aim for **4-7 criteria**.

3. **Self-contained.** Each criterion must be independently checkable by reading
   the evidence — no cross-references between criteria, no domain knowledge
   required beyond what is in the question itself.

4. **Binary positive form.** Every criterion is a positive statement; the
   verifier scores it 1.0 (satisfied) / 0.5 (partial) / 0.0 (unsatisfied). Do
   NOT phrase criteria as negatives or "pitfalls".

# Output format

Output **valid YAML only**, no prose. Each criterion has:

- `name`: snake_case identifier (≤ 30 chars)
- `description`: one positive sentence stating what evidence must show
- `score_1`: what perfect satisfaction looks like
- `score_half`: what partial satisfaction looks like
- `score_0`: what total failure looks like
- `weight`: float in [0.1, 1.0] — relative importance to answering this question
- `failure_repair_action`: one of `refine_query` / `asr_match` / `time_sorted` /
  `dense_sample` / `loose_verify` / `broadcast`

```yaml
criteria:
  - name: <snake_case>
    description: "..."
    score_1: "..."
    score_half: "..."
    score_0: "..."
    weight: 0.0
    failure_repair_action: refine_query
  - ...
```

Do not output a `scoring_rule` or `sufficient_threshold` — those are set
downstream during distillation.
