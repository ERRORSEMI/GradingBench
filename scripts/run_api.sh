#!/usr/bin/env bash
# API 推理 + 可选评估
# Usage: bash scripts/run_api.sh L2 --model glm-4.6v --need_answer False
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LEVEL="${1:?usage: run_api.sh L1|L2|L3 [--model ...]}"
shift
pigai_init_shell "$LEVEL"

MODEL="${MODEL:-glm-4.6v}"
NEED_ANSWER="${NEED_ANSWER:-False}"
RUN_EVAL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --need_answer) NEED_ANSWER="$2"; shift 2 ;;
        --no_eval) RUN_EVAL=0; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== [${PIGAI_LEVEL}] API inference model=$MODEL need_answer=$NEED_ANSWER ==="

python -m gradingbench.inference.api \
    --level "$PIGAI_LEVEL" \
    --model "$MODEL" \
    --subjects chinese,science,liberal_arts,math,english \
    --need_answer "$NEED_ANSWER"

if [[ "$RUN_EVAL" -eq 1 ]]; then
    bash "${PIGAI_SHELL_DIR}/evaluator.sh" "$PIGAI_LEVEL" \
        --model "$MODEL" --need_answer "$NEED_ANSWER"
fi
