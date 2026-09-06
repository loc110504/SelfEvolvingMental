#!/usr/bin/env bash
# Runs all 3 simple PHQ-8 baselines (direct / cot / fewshot) on dev and test,
# then prints + saves a comparison table across them (and, if you list its
# summary path(s) in EXTRA_SUMMARIES below, against the proposed Grounded
# Interview Loop method's run_batch_eval.py output).
#
# Usage:
#   ./scripts/run_all_baselines.sh                 # all 3 methods x dev+test
#   ./scripts/run_all_baselines.sh direct           # one method, dev+test
#   ./scripts/run_all_baselines.sh direct dev       # one method, one split
#
# Edit MODEL_NAME / API_BASE_URL below if you point this at a different
# server or model.

set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
API_BASE_URL="http://127.0.0.1:8000/v1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="results/evaluations/baselines"
ALL_METHODS=(direct cot fewshot)
ALL_SPLITS=(dev test)

# Proposed-method summaries to fold into the final comparison table (optional).
# Example:
#   EXTRA_SUMMARIES=(results/evaluations/qwen2.5_14b_memory_only_v2/batch_summary_*.json)
EXTRA_SUMMARIES=()

if [ $# -eq 0 ]; then
    METHODS=("${ALL_METHODS[@]}")
    SPLITS=("${ALL_SPLITS[@]}")
elif [ $# -eq 1 ]; then
    METHODS=("$1")
    SPLITS=("${ALL_SPLITS[@]}")
else
    METHODS=("$1")
    SPLITS=("$2")
fi

for method in "${METHODS[@]}"; do
    for split in "${SPLITS[@]}"; do
        echo "=== Baseline '$method' on $split ==="
        python3 scripts/run_baseline_eval.py \
            --method "$method" \
            --data-dir "data/processed_daic_woz/$split" \
            --model-name "$MODEL_NAME" \
            --api-base-url "$API_BASE_URL" \
            --output-dir "$OUTPUT_DIR" \
            --skip-existing
    done
done

echo ""
echo "=== Comparison table ==="
SUMMARIZE_ARGS=(--baseline-dir "$OUTPUT_DIR" --output-csv "$OUTPUT_DIR/comparison.csv")
if [ "${#EXTRA_SUMMARIES[@]}" -gt 0 ]; then
    SUMMARIZE_ARGS+=(--extra-summary "${EXTRA_SUMMARIES[@]}")
fi
python3 scripts/summarize_baselines.py "${SUMMARIZE_ARGS[@]}"
