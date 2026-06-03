#!/bin/bash
# VEIL Ablation Experiments — Run 5 experiments in parallel on 2 GPUs
# GPU1: singlequery, no_rubric_judge, ignore_verifier (queue)
# GPU2: oracle, strict_dedup, loose_query_dedup (queue)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Config file
CONFIG="${1:-configs/videomme_memory_bank.yaml}"
VLM_API_GPU1="${2:-http://localhost:8778}"
VLM_API_GPU2="${3:-http://localhost:8779}"
BGE_GPU1="${4:-cuda:1}"
BGE_GPU2="${5:-cuda:2}"

# Output directory
OUT_DIR="outputs/results/videommeL"
mkdir -p "$OUT_DIR"

echo "===================================================================="
echo "VEIL Ablation Experiments — Parallel (2 GPU tracks, 2 separate vLLM)"
echo "===================================================================="
echo "Config: $CONFIG"
echo "Output: $OUT_DIR"
echo "Track1 vLLM (GPU1): $VLM_API_GPU1"
echo "Track2 vLLM (GPU2): $VLM_API_GPU2"
echo "BGE GPU1: $BGE_GPU1"
echo "BGE GPU2: $BGE_GPU2"
echo ""
echo "Track 1 (GPU1): veil_27b_singlequery → veil_27b_no_rubric_judge → veil_27b_ignore_verifier"
echo "Track 2 (GPU2): veil_27b_oracle → veil_27b_strict_dedup → veil_27b_high_query_threshold"
echo ""

# Track 1: GPU1 (3 ablations in sequence)
(
  echo "[Track1-1/3] Ablation: veil_27b_singlequery"
  PYTHONPATH=. python experiments/veil_27b_singlequery.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU1" \
      --llm-api-url "$VLM_API_GPU1" \
      --bge-gpu "$BGE_GPU1" \
      --workers 1

  echo ""
  echo "[Track1-2/3] Ablation: veil_27b_no_rubric_judge"
  PYTHONPATH=. python experiments/veil_27b_no_rubric_judge.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU1" \
      --llm-api-url "$VLM_API_GPU1" \
      --bge-gpu "$BGE_GPU1" \
      --workers 1

  echo ""
  echo "[Track1-3/3] Ablation: veil_27b_ignore_verifier"
  PYTHONPATH=. python experiments/veil_27b_ignore_verifier.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU1" \
      --llm-api-url "$VLM_API_GPU1" \
      --bge-gpu "$BGE_GPU1" \
      --workers 1

  echo ""
  echo "==== Track 1 (GPU1) Complete ===="
) &
TRACK1_PID=$!

# Track 2: GPU2 (oracle + 2 dedup ablations)
(
  echo "[Track2-0/3] Baseline: veil_27b_oracle (reference)"
  PYTHONPATH=. python experiments/veil_27b_oracle.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU2" \
      --llm-api-url "$VLM_API_GPU2" \
      --bge-gpu "$BGE_GPU2" \
      --workers 1

  echo ""
  echo "[Track2-1/3] Ablation: veil_27b_strict_dedup — stricter evidence dedup (0.85→0.90)"
  PYTHONPATH=. python experiments/veil_27b_strict_dedup.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU2" \
      --llm-api-url "$VLM_API_GPU2" \
      --bge-gpu "$BGE_GPU2" \
      --workers 1

  echo ""
  echo "[Track2-2/3] Ablation: veil_27b_high_query_threshold — higher dedup threshold (0.9→0.99)"
  PYTHONPATH=. python experiments/veil_27b_high_query_threshold.py \
      --config "$CONFIG" \
      --vlm-api-url "$VLM_API_GPU2" \
      --llm-api-url "$VLM_API_GPU2" \
      --bge-gpu "$BGE_GPU2" \
      --workers 1

  echo ""
  echo "==== Track 2 (GPU2) Complete ===="
) &
TRACK2_PID=$!

echo "Both tracks started in background..."
echo "  Track 1 PID: $TRACK1_PID"
echo "  Track 2 PID: $TRACK2_PID"
echo ""

# Wait for both to finish
wait $TRACK1_PID
TRACK1_STATUS=$?

wait $TRACK2_PID
TRACK2_STATUS=$?

echo ""
echo "===================================================================="
echo "All experiments completed!"
if [ $TRACK1_STATUS -eq 0 ] && [ $TRACK2_STATUS -eq 0 ]; then
  echo "✓ Both tracks completed successfully"
else
  echo "✗ Some tracks failed (Track1: $TRACK1_STATUS, Track2: $TRACK2_STATUS)"
fi
echo "Results in: $OUT_DIR"
echo "===================================================================="
echo ""
echo "Compare results with:"
echo "  python scripts/compare_ablations.py"
