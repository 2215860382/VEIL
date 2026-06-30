#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home2/ycj/miniconda3/envs/veil/bin/python}"
LLM_API_URL="${LLM_API_URL:-http://127.0.0.1:8003,http://127.0.0.1:8004}"
RESULT="${RESULT:-outputs/debug/mf_resized_planslim_full_900.jsonl}"
MEMORY_DIR="${MEMORY_DIR:-outputs/memory/videomme_multiframe}"
OUT_ROOT="${OUT_ROOT:-outputs/results/intermediate_goods/overnight_20260627_full720}"

mkdir -p "$OUT_ROOT"

if [ ! -f "$OUT_ROOT/round1_type_level_rubric.yaml" ]; then
  echo "[round1] aggregating type-level rubric"
  "$PYTHON" -m src.generate_rubric.construct.aggregate_type_rubric \
    --criteria "$OUT_ROOT/round1_pair_criteria.jsonl" \
    --llm-api-url "$LLM_API_URL" \
    --out "$OUT_ROOT/round1_type_level_rubric.yaml"
fi

if [ ! -f "$OUT_ROOT/round1_backtest_report.json" ]; then
  echo "[round1] backtest"
  "$PYTHON" -m src.generate_rubric.construct.backtest_report \
    --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
    --chains "$OUT_ROOT/round1_chains.jsonl" \
    --pairs "$OUT_ROOT/round1_pairs.jsonl" \
    --criteria "$OUT_ROOT/round1_pair_criteria.jsonl" \
    --out "$OUT_ROOT/round1_backtest_report.json"
fi

if [ ! -f "$OUT_ROOT/round2_pair_criteria.jsonl" ]; then
  echo "[round2] pairwise extraction with round1 rubric"
  "$PYTHON" -m src.generate_rubric.construct.extract_pair_criteria \
    --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
    --chains "$OUT_ROOT/round1_chains.jsonl" \
    --pairs "$OUT_ROOT/round1_pairs.jsonl" \
    --rubric-path "$OUT_ROOT/round1_type_level_rubric.yaml" \
    --llm-api-url "$LLM_API_URL" \
    --workers 2 \
    --out "$OUT_ROOT/round2_pair_criteria.jsonl"
fi

if [ ! -f "$OUT_ROOT/round2_type_level_rubric.yaml" ]; then
  echo "[round2] aggregating type-level rubric"
  "$PYTHON" -m src.generate_rubric.construct.aggregate_type_rubric \
    --criteria "$OUT_ROOT/round2_pair_criteria.jsonl" \
    --llm-api-url "$LLM_API_URL" \
    --out "$OUT_ROOT/round2_type_level_rubric.yaml"
fi

if [ ! -f "$OUT_ROOT/round2_backtest_report.json" ]; then
  echo "[round2] backtest"
  "$PYTHON" -m src.generate_rubric.construct.backtest_report \
    --dev "$OUT_ROOT/splits/dev_questions.jsonl" \
    --chains "$OUT_ROOT/round1_chains.jsonl" \
    --pairs "$OUT_ROOT/round1_pairs.jsonl" \
    --criteria "$OUT_ROOT/round2_pair_criteria.jsonl" \
    --out "$OUT_ROOT/round2_backtest_report.json"
fi

echo "resume rubric construction artifacts: $OUT_ROOT"
