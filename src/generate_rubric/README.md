# generate_rubric — Evidence-Sufficiency Rubric Construction

Code-only package that builds long-video multiple-choice-QA **evidence-sufficiency**
rubrics for the VEIL verifier. A rubric scores whether a retrieved evidence chain is
sufficient to **verify or exclude each option** — not answer quality, not option
support. (A false option can score high when the evidence is enough to rule it out.)

## Where things live

| What | Where |
|------|-------|
| Pipeline code (this package) | `src/generate_rubric/construct/*.py` |
| Runnable shell scripts | `scripts/*.sh` |
| Final runtime rubrics (tracked) | `outputs/rubric/*.yaml` |
| Intermediate construction artifacts (gitignored) | `outputs/results/intermediate_goods/` |

Final rubrics in `outputs/rubric/`:
- `direct_answer_generated_v2.yaml` — default runtime rubric (distilled, 7 weighted general criteria).
- `tsqf_handwritten_legacy.yaml` — handwritten TSQF baseline.
- `round1_type_level_rubric.yaml` — round-1 pairwise-constructed rubric.
- `round2_merged_runtime.yaml` — round1 ∪ round2 merged (main constructed rubric).
- `round2_weighted_runtime.yaml` — round2 + LLM criterion weights (1–5) + groundedness guard.

The verifier loads `outputs/rubric/direct_answer_generated_v2.yaml` by default; override
per run with `VEIL_RUBRIC_PATH=<path>` (each run records the rubric path + sha1 in its
`*.meta.json`).

## Method

- **OnlineRubrics** (pairwise elicitation): start from an existing coarse rubric, compare
  two evidence chains, extract only the differences not already covered.
- **Chasing the Tail**: prioritize differences between *high-quality* chains
  (good-vs-great / great-vs-excellent), not just weak-vs-good.

Pipeline stages (`construct/`):
`sample_dev` → `chains_from_result` → `make_pairs` → `extract_pair_criteria`
→ `aggregate_type_rubric` → (`weight_rubric`) → `backtest_report`.

`aggregate_type_rubric` supports `--render-only` (re-render a complete `.state.json`
without any LLM call), `--base-rubric` (merge a round's delta onto a base rubric), and
`--agg-workers` (aggregate question types concurrently).

## Run

```bash
# Two-round construction (writes to outputs/results/intermediate_goods/overnight_*)
RESULT=outputs/debug/mf_resized_planslim_full_900.jsonl \
MEMORY_DIR=outputs/memory/videomme_multiframe \
LLM_API_URL=http://127.0.0.1:8003,http://127.0.0.1:8004 \
bash scripts/run_two_rounds.sh

# Optional: weight + anti-gaming guard
PYTHONPATH=. python -m src.generate_rubric.construct.weight_rubric \
  --rubric outputs/rubric/round2_merged_runtime.yaml \
  --out outputs/rubric/round2_weighted_runtime.yaml
```

## Result (VideoMME, this build)

On held-out val-90 / test-90 / full-900, the constructed rubrics, the distilled `v2`,
and the handwritten `tsqf` baseline all land within noise (~79–80% on 180 clean
questions; 74% on 900). Diagnostics (oracle perfect-planner ceiling 76.4%, keyframe
k=64 probe, and a rubric-score-vs-prediction analysis showing the rubric changes
verifier scoring on 141/180 questions but flips only 11/180 final answers) indicate the
system is **answerer-bound**: rubric design is correct and active but is not the accuracy
lever at this scale.

Two answerer-prompt interventions were tried and both falsified:

1. **Keyword-routed question-type guidance** (negation / synopsis / temporal) hurt both
   held-out splits in the same direction (val 81.1%→73.3%, test 78.9%→75.6%; 12
   regressions vs 2 gains). Reverted.
2. **Chain-of-thought answerer** (reason briefly over evidence timestamps, then emit the
   letter) looked marginally positive on 180 (val −1, test +4, net +3) — but that was
   small-sample noise. On **dev-720** it was significantly *negative*: 72.5%→69.6%
   (−2.9 pt, 31 gained / 52 lost, McNemar p=0.021). Reverted. Forcing the answerer to
   reason over the lossy memory summary makes it argue itself out of correct intuitions.

Method lesson: **only ≥720-question results are trustworthy here**; 180-split rankings
flip as noise (this is why the constructed rubrics looked split-dependent). The answerer
is near its ceiling on the given evidence — gains require *better evidence* (retrieval
recall / richer memory captions), not answerer-prompt steering. A text-only full-context
probe (feed all chunk texts, no retrieval) was net +8 on the 180-split, suggesting a real
recall gap, but that too needs 720-scale confirmation before being trusted.
