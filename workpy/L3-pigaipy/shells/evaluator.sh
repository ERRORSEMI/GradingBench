#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/../scripts/lib.sh"
pigai_init_shell "L3"

DEFAULT_MODEL="${DEFAULT_MODEL:-glm-4.6v}"
MODEL="$DEFAULT_MODEL"
NEED_ANSWER="False"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --need_answer) NEED_ANSWER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== [L3] model=$MODEL need_answer=$NEED_ANSWER ==="

python "${PIGAI_LEVEL_DIR}/stage1_generate_raw_results.py" \
    --model "$MODEL" --need_answer "$NEED_ANSWER"

python "${PIGAI_LEVEL_DIR}/stage2_filter_with_qwen.py" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-2}" \
    --model "$MODEL" --need_answer "$NEED_ANSWER" \
    --model_path "${PIGAI_FILTER_MODEL_PATH}"

python "${PIGAI_LEVEL_DIR}/stage3_calculate_metrics.py" \
    --model "$MODEL" --need_answer "$NEED_ANSWER"

echo "=== Done ==="
echo "Metrics saved under results/predictions/L3/"
