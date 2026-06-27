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
