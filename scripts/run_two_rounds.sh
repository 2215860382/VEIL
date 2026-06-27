#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
RESULT="${RESULT:-outputs/debug/mf_resized_planslim_full_900.jsonl}"
MEMORY_DIR="${MEMORY_DIR:-outputs/memory/videomme_multiframe}"
LLM_API_URL="${LLM_API_URL:-http://127.0.0.1:8003,http://127.0.0.1:8004}"
OUT_ROOT="${OUT_ROOT:-outputs/results/intermediate_goods/overnight_$(date +%Y%m%d_%H%M%S)}"
N_DEV="${N_DEV:-720}"
N_VAL="${N_VAL:-90}"
N_TEST="${N_TEST:-90}"
WORKERS="${WORKERS:-2}"
mkdir -p "$OUT_ROOT"

echo "[round0] selecting splits"
"$PYTHON" -m src.generate_rubric.construct.sample_dev \
  --config configs/veil.yaml \
  --result "$RESULT" \
  --n-dev "$N_DEV" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --out-dir "$OUT_ROOT/splits"

echo "[round1] building chains/pairs"
"$PYTHON" -m src.generate_rubric.construct.chains_from_result \
  --config configs/veil.yaml \
  --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
  --result "$RESULT" \
  --memory-dir "$MEMORY_DIR" \
  --out "$OUT_ROOT/round1_chains.jsonl"
"$PYTHON" -m src.generate_rubric.construct.make_pairs \
  --chains "$OUT_ROOT/round1_chains.jsonl" \
  --out "$OUT_ROOT/round1_pairs.jsonl"

echo "[round1] pairwise extraction"
"$PYTHON" -m src.generate_rubric.construct.extract_pair_criteria \
  --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
  --chains "$OUT_ROOT/round1_chains.jsonl" \
  --pairs "$OUT_ROOT/round1_pairs.jsonl" \
  --llm-api-url "$LLM_API_URL" \
  --workers "$WORKERS" \
  --out "$OUT_ROOT/round1_pair_criteria.jsonl"
"$PYTHON" -m src.generate_rubric.construct.aggregate_type_rubric \
  --criteria "$OUT_ROOT/round1_pair_criteria.jsonl" \
  --llm-api-url "$LLM_API_URL" \
  --out "$OUT_ROOT/round1_type_level_rubric.yaml"
"$PYTHON" -m src.generate_rubric.construct.backtest_report \
  --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
  --chains "$OUT_ROOT/round1_chains.jsonl" \
  --pairs "$OUT_ROOT/round1_pairs.jsonl" \
  --criteria "$OUT_ROOT/round1_pair_criteria.jsonl" \
  --out "$OUT_ROOT/round1_backtest_report.json"

echo "[round2] pairwise extraction with round1 rubric"
"$PYTHON" -m src.generate_rubric.construct.extract_pair_criteria \
  --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
  --chains "$OUT_ROOT/round1_chains.jsonl" \
  --pairs "$OUT_ROOT/round1_pairs.jsonl" \
  --rubric-path "$OUT_ROOT/round1_type_level_rubric.yaml" \
  --llm-api-url "$LLM_API_URL" \
  --workers "$WORKERS" \
  --out "$OUT_ROOT/round2_pair_criteria.jsonl"
"$PYTHON" -m src.generate_rubric.construct.aggregate_type_rubric \
  --criteria "$OUT_ROOT/round2_pair_criteria.jsonl" \
  --llm-api-url "$LLM_API_URL" \
  --out "$OUT_ROOT/round2_type_level_rubric.yaml"
"$PYTHON" -m src.generate_rubric.construct.backtest_report \
  --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
  --chains "$OUT_ROOT/round1_chains.jsonl" \
  --pairs "$OUT_ROOT/round1_pairs.jsonl" \
  --criteria "$OUT_ROOT/round2_pair_criteria.jsonl" \
  --out "$OUT_ROOT/round2_backtest_report.json"

echo "overnight rubric construction artifacts: $OUT_ROOT"
