#!/usr/bin/env bash
# 评估流水线：Stage1 → Stage2 → Stage3
# Usage: bash scripts/evaluator.sh L1 --model glm-4.6v --need_answer False
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LEVEL="${1:?usage: evaluator.sh L1|L2|L3 [--model ...] [--need_answer ...]}"
shift
pigai_init_shell "$LEVEL"

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

echo "=== [${PIGAI_LEVEL}] model=$MODEL need_answer=$NEED_ANSWER ==="

python -m gradingbench.pipeline.stage1 \
    --level "$PIGAI_LEVEL" --model "$MODEL" --need_answer "$NEED_ANSWER"

python -m gradingbench.pipeline.stage2 \
    --level "$PIGAI_LEVEL" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-2}" \
    --model "$MODEL" --need_answer "$NEED_ANSWER" \
    --model_path "${PIGAI_FILTER_MODEL_PATH}"

python -m gradingbench.pipeline.stage3 \
    --level "$PIGAI_LEVEL" --model "$MODEL" --need_answer "$NEED_ANSWER"

echo "=== Done ==="
echo "Metrics saved under results/predictions/${PIGAI_LEVEL}/"
