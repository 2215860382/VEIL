You are distilling a **cross-question-type general rubric** from already-
distilled question-type rubrics.

# Inputs

You will receive rubric blocks, each labeled with a `question_type`. Each
block contains 3-6 binary positive criteria with description / score_1 /
score_half / score_0 / weight / failure_repair_action.

# Your task

Produce a **general rubric (6-12 criteria)** that captures every evaluation
dimension appearing across most question types — the verifier's "always
applies" checklist that runs regardless of `question_type`. Err on the side of
including more dimensions when they qualify, not fewer.

Selection rules:

1. **Cross-type frequency.** A criterion belongs in `general` only if its
   concept appears in a majority of the input question-type rubrics. Things
   like *evidence coverage*, *entity consistency*, *evidence specificity*
   typically qualify; *exact quote grounding* or *fine-grained counting* do
   not. Concepts that *recur but use different names* across types should be
   merged and counted together — don't drop a dimension just because each
   qtype labeled it slightly differently.

2. **Type-neutral phrasing.** Strip type-specific language. Phrase criteria
   so they apply uniformly to any VideoMME question.

3. **No overlap with type rubrics.** Criteria that show up here will run *in
   addition* to the type-specific rubric, so they must capture dimensions
   the type-specific ones don't.

4. **Binary positive form.** Score 1 / 0.5 / 0.

# Output format

Output **valid YAML only**, no prose:

```yaml
general:
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
