#!/usr/bin/env bash
# vLLM 推理 + 可选评估
# Usage: bash scripts/run_vllm.sh L1 --model Qwen2.5-VL-7B-Instruct --need_answer False
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LEVEL="${1:?usage: run_vllm.sh L1|L2|L3 [--model ...] [--no_eval]}"
shift
pigai_init_shell "$LEVEL"

MODEL="${MODEL:-Qwen2.5-VL-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/qwen/Qwen2.5-VL-7B-Instruct}"
NEED_ANSWER="${NEED_ANSWER:-False}"
RUN_EVAL=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export SIZE_FACTOR="${SIZE_FACTOR:-8}"
export MIN_PIXELS="${MIN_PIXELS:-$((224 * 224))}"
export MAX_PIXELS="${MAX_PIXELS:-$((2200 * 28 * 28))}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --need_answer) NEED_ANSWER="$2"; shift 2 ;;
        --no_eval) RUN_EVAL=0; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== [${PIGAI_LEVEL}] vLLM inference model=$MODEL need_answer=$NEED_ANSWER ==="

python -m gradingbench.inference.vllm \
    --level "$PIGAI_LEVEL" \
    --model_path "$MODEL_PATH" \
    --model "$MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --subjects chinese,science,liberal_arts,math,english \
    --need_answer "$NEED_ANSWER"

if [[ "$RUN_EVAL" -eq 1 ]]; then
    bash "${PIGAI_SHELL_DIR}/evaluator.sh" "$PIGAI_LEVEL" \
        --model "$MODEL" --need_answer "$NEED_ANSWER"
fi
