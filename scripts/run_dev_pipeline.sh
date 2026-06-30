#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT="${RESULT:-outputs/debug/mf_resized_planslim_full_900.jsonl}"
MEMORY_DIR="${MEMORY_DIR:-outputs/memory/videomme_multiframe}"
OUT_DIR="${OUT_DIR:-outputs/results/intermediate_goods/iter$(date +%s)}"
LLM_API_URL="${LLM_API_URL:-http://127.0.0.1:8003,http://127.0.0.1:8004}"
N_DEV="${N_DEV:-720}"
N_VAL="${N_VAL:-90}"
N_TEST="${N_TEST:-90}"
WORKERS="${WORKERS:-2}"
MAX_PAIRS="${MAX_PAIRS:-0}"
PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"

mkdir -p "$OUT_DIR"
MAX_PAIR_ARGS=()
if [ "$MAX_PAIRS" != "0" ]; then
  MAX_PAIR_ARGS=(--max-pairs "$MAX_PAIRS")
fi

"$PYTHON" -m src.generate_rubric.construct.sample_dev \
  --config configs/veil.yaml \
  --result "$RESULT" \
  --n-dev "$N_DEV" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --out-dir "$OUT_DIR"

"$PYTHON" -m src.generate_rubric.construct.chains_from_result \
  --config configs/veil.yaml \
  --dev "$OUT_DIR/dev_questions.jsonl" \
  --result "$RESULT" \
  --memory-dir "$MEMORY_DIR" \
  --out "$OUT_DIR/chains.jsonl"

"$PYTHON" -m src.generate_rubric.construct.make_pairs \
  --chains "$OUT_DIR/chains.jsonl" \
  --out "$OUT_DIR/pairs.jsonl"

"$PYTHON" -m src.generate_rubric.construct.extract_pair_criteria \
  --dev "$OUT_DIR/dev_questions.jsonl" \
  --chains "$OUT_DIR/chains.jsonl" \
  --pairs "$OUT_DIR/pairs.jsonl" \
  --llm-api-url "$LLM_API_URL" \
  --workers "$WORKERS" \
  "${MAX_PAIR_ARGS[@]}" \
  --out "$OUT_DIR/pair_criteria.jsonl"

"$PYTHON" -m src.generate_rubric.construct.aggregate_type_rubric \
  --criteria "$OUT_DIR/pair_criteria.jsonl" \
  --llm-api-url "$LLM_API_URL" \
  --out "$OUT_DIR/type_level_rubric.yaml"

"$PYTHON" -m src.generate_rubric.construct.backtest_report \
  --dev "$OUT_DIR/dev_questions.jsonl" \
  --chains "$OUT_DIR/chains.jsonl" \
  --pairs "$OUT_DIR/pairs.jsonl" \
  --criteria "$OUT_DIR/pair_criteria.jsonl" \
  --out "$OUT_DIR/backtest_report.json"

echo "rubric construction artifacts: $OUT_DIR"
