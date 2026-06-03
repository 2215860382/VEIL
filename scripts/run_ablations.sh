#!/bin/bash
# VEIL Ablation Experiments — Run all P1 priority ablations

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Config file
CONFIG="${1:-configs/videomme_memory_bank.yaml}"
VLM_API="${2:-http://localhost:8000}"
LLM_API="${3:-http://localhost:8001}"
BGE_GPU="${4:-cuda:3}"

# Output directory
OUT_DIR="outputs/results/videommeL"
mkdir -p "$OUT_DIR"

echo "===================================================================="
echo "VEIL Ablation Experiments — 4 Core Dimensions"
echo "===================================================================="
echo "Config: $CONFIG"
echo "Output: $OUT_DIR"
echo "VLM API: $VLM_API"
echo "LLM API: $LLM_API"
echo "BGE GPU: $BGE_GPU"
echo ""

# Baseline (for reference)
echo "[0/5] Running baseline: veil_27b (reference — 71.56%)"
PYTHONPATH=. python experiments/veil_27b.py \
    --config "$CONFIG" \
    --vlm-api-url "$VLM_API" \
    --llm-api-url "$LLM_API" \
    --bge-gpu "$BGE_GPU" \
    --workers 1

# Ablation 1: Single query (no decomposition)
echo ""
echo "[2/5] Dimension 1: veil_27b_singlequery — no decomposition, use original question"
PYTHONPATH=. python experiments/veil_27b_singlequery.py \
    --config "$CONFIG" \
    --vlm-api-url "$VLM_API" \
    --llm-api-url "$LLM_API" \
    --bge-gpu "$BGE_GPU" \
    --workers 1

# Ablation 2: No rubric judgment
echo ""
echo "[3/5] Dimension 2: veil_27b_no_rubric_judge — disable rubric scoring"
PYTHONPATH=. python experiments/veil_27b_no_rubric_judge.py \
    --config "$CONFIG" \
    --vlm-api-url "$VLM_API" \
    --llm-api-url "$LLM_API" \
    --bge-gpu "$BGE_GPU" \
    --workers 1

# Ablation 3: Strict evidence deduplication
echo ""
echo "[4/5] Dimension 3: veil_27b_strict_dedup — stricter evidence dedup (0.85→0.90)"
PYTHONPATH=. python experiments/veil_27b_strict_dedup.py \
    --config "$CONFIG" \
    --vlm-api-url "$VLM_API" \
    --llm-api-url "$LLM_API" \
    --bge-gpu "$BGE_GPU" \
    --workers 1

# Ablation 4: Planner ignores Verifier signal
echo ""
echo "[5/5] Dimension 4: veil_27b_ignore_verifier — Planner ignores Verifier feedback"
PYTHONPATH=. python experiments/veil_27b_ignore_verifier.py \
    --config "$CONFIG" \
    --vlm-api-url "$VLM_API" \
    --llm-api-url "$LLM_API" \
    --bge-gpu "$BGE_GPU" \
    --workers 1

echo ""
echo "===================================================================="
echo "All ablations completed!"
echo "Results in: $OUT_DIR"
echo "===================================================================="
echo ""
echo "Compare results with:"
echo "  python scripts/compare_ablations.py"
