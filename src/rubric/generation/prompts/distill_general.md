You are distilling a **cross-question-type general evidence-requirement rubric**
from already-distilled question-type requirement blocks.

# Inputs

You will receive requirement blocks labeled by `question_type`. Each block has
3-6 evidence requirements.

# Task

Produce a general rubric containing evidence requirements that should run for
every VideoMME question before type-specific requirements.

The verifier uses this rubric with coverage-first semantics:
- incomplete required evidence -> option stance is unknown
- complete required evidence -> option can be judged support or refute

# Selection Rules

1. **Cross-type frequency.** Include a requirement only when its concept appears
   across many question types. Target grounding and evidence specificity usually
   qualify; exact quote grounding or fine-grained counting usually does not.

2. **Type-neutral phrasing.** Phrase each requirement so it applies uniformly to
   any question and any option.

3. **No overlap with type-specific requirements.** General requirements should
   cover universal prerequisites. Leave specialized needs to qtype templates.

4. **Compact.** Output 1-4 general requirements. Too many general requirements
   will make the verifier overly strict.

# Output Format

Output valid YAML only, no prose:

```yaml
general:
  evidence_requirements:
    - id: <snake_case_id>
      description: "Concrete universal fact that must be covered before judging options."
      required: true
      weight: 0.0
      modality: visual|dialogue|temporal|ocr|global|multimodal
      repair_action: refine_query
  decision_policy:
    support_confidence_threshold: 0.80
    refute_confidence_threshold: 0.70
    require_all_options_resolved: true
```
