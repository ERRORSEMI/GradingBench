#!/usr/bin/env bash
# Unified entry: L1/L2/L3 × answer-free/answer-based
# Usage:
#   bash scripts/evaluate.sh L1 false --model Qwen2.5-VL-7B-Instruct
#   bash scripts/evaluate.sh all true --model glm-4.6v

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

LEVEL="${1:?用法: evaluate.sh <L1|L2|L3|all> <true|false> [--model M]}"
NEED_ANSWER="${2:?用法: evaluate.sh <L1|L2|L3|all> <true|false>}"
shift 2 || true

MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

run_one() {
  local lv="$1"
  [[ -n "$MODEL" ]] || { echo "需指定 --model"; exit 1; }
  bash "${PIGAI_REPO_ROOT}/scripts/evaluator.sh" "$lv" \
    --model "$MODEL" --need_answer "$NEED_ANSWER"
}

if [[ "$LEVEL" == "all" ]]; then
  for lv in L1 L2 L3; do
    echo "########## ${lv} need_answer=${NEED_ANSWER} ##########"
    run_one "$lv"
  done
else
  run_one "$LEVEL"
fi
