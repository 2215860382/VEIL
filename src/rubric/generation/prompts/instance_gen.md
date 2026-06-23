You are designing **per-question evidence requirements** for a video QA verifier.

The verifier must first decide whether the retrieved evidence is complete enough
to judge each answer option. Only after coverage is complete may it label an
option as supported or refuted.

# Inputs

You will receive:
- **question**: the natural-language question
- **candidates**: four answer options A/B/C/D
- **question_type**: VideoMME category

Do not assume the correct answer is known. Do not write requirements that reveal
or prefer any specific option. Requirements must be answer-agnostic and usable
to verify every option.

# Design Principles

1. **Coverage first.** Each requirement names a concrete fact that must be
   present before judging support/refute. If the fact is missing, the verifier
   must output unknown for the affected option.

2. **Option-comparable.** Requirements must apply to every answer option, not
   only one option. Phrase them as evidence needs, not conclusions.

3. **Concrete retrieval target.** Each requirement should identify what the
   retriever should look for: target entity, relevant action/state, temporal
   anchor, visual text, count scope, spoken phrase, spatial relation, or global
   coverage.

4. **Minimal and non-overlapping.** Produce 3-6 requirements. Avoid redundant
   requirements that would trigger the same retrieval query.

# Repair Actions

Choose one of two repair actions:
- `refine_query`: issue focused semantic retrieval queries for specific missing facts
- `broadcast`: uniformly sample broader unseen video segments when the missing
  evidence requires global or timeline-wide coverage

# Output Format

Output valid YAML only, no prose:

```yaml
evidence_requirements:
  - id: <snake_case_id>
    description: "Concrete fact that must be covered before judging options."
    required: true
    weight: 0.0
    modality: visual|dialogue|temporal|ocr|global|multimodal
    repair_action: refine_query
```

Use weights in [0.1, 1.0]. Higher weight means the requirement is more central
to answering the question.
