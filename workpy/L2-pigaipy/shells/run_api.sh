#!/usr/bin/env bash
# Hosted VL API inference template for L2.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/../scripts/lib.sh"
pigai_init_shell "L2"

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

echo "=== [L2] API inference model=$MODEL need_answer=$NEED_ANSWER ==="

python "${PIGAI_LEVEL_DIR}/main_modelpigai_api.py" \
    --model "$MODEL" \
    --subjects chinese,science,liberal_arts,math,english \
    --need_answer "$NEED_ANSWER"

if [[ "$RUN_EVAL" -eq 1 ]]; then
    bash "${PIGAI_SHELL_DIR}/evaluator.sh" --model "$MODEL" --need_answer "$NEED_ANSWER"
fi
