#!/usr/bin/env bash
# Runs the Grounded Interview Loop evaluation (Plan_Improve.md Phase 1) in
# the correct split order:
#   Step 2 — tune/sanity-check on `train` (never tune on dev/test)
#   Step 3 — official comparison run on `dev` against the baseline
#            (qwen2.5_14b_memory_only: MAE=5.0, Pearson r=0.50)
#
# Usage:
#   ./scripts/run_grounded_eval.sh              # runs both steps
#   ./scripts/run_grounded_eval.sh train         # train step only
#   ./scripts/run_grounded_eval.sh dev           # dev step only
#
# Edit MODEL_NAME / API_BASE_URL below if you point this at a different
# server or model.

set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
API_BASE_URL="http://127.0.0.1:8000/v1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_OUTPUT_DIR="results/evaluations/qwen2.5_14b_memory_only_train_v2"
DEV_OUTPUT_DIR="results/evaluations/qwen2.5_14b_memory_only_v2"

run_train() {
    echo "=== Step 2: tuning/sanity-check run on train ==="
    python3 scripts/run_batch_eval.py \
        --data-dir data/processed_daic_woz/train \
        --model-name "$MODEL_NAME" \
        --api-base-url "$API_BASE_URL" \
        --enable-memory-update \
        --output-dir "$TRAIN_OUTPUT_DIR" \
        --skip-existing
}

run_dev() {
    echo "=== Step 3: official comparison run on dev ==="
    python3 scripts/run_batch_eval.py \
        --data-dir data/processed_daic_woz/dev \
        --model-name "$MODEL_NAME" \
        --api-base-url "$API_BASE_URL" \
        --enable-memory-update \
        --output-dir "$DEV_OUTPUT_DIR" \
        --skip-existing
}

case "${1:-all}" in
    train) run_train ;;
    dev) run_dev ;;
    all)
        run_train
        run_dev
        ;;
    *)
        echo "Usage: $0 [train|dev|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Done. Compare against baseline (results/evaluations/qwen2.5_14b_memory_only):"
echo "  Train run:  $TRAIN_OUTPUT_DIR"
echo "  Dev run:    $DEV_OUTPUT_DIR (MAE/RMSE/Pearson r/F1 + evidence diagnostics printed above)"
