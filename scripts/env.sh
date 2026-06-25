#!/usr/bin/env bash
# 开源版环境变量。用法: source scripts/env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PIGAI_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PIGAI_SRC_ROOT="${PIGAI_REPO_ROOT}/src"
export PIGAI_DATA_ROOT="${PIGAI_DATA_ROOT:-${PIGAI_REPO_ROOT}/data}"
export PIGAI_RESULTS_ROOT="${PIGAI_RESULTS_ROOT:-${PIGAI_REPO_ROOT}/results}"
export PYTHONPATH="${PIGAI_SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# GradingBench dataset (Hugging Face)
export HF_DATASET_REPO="${HF_DATASET_REPO:-ERRORSEMI/GradingBench}"

# Open-source model weights root (see scripts/run_vllm.sh)
export MODEL_ROOT="${MODEL_ROOT:-/path/to/models}"

# 文本筛选模型 (stage2)
export PIGAI_FILTER_MODEL_PATH="${PIGAI_FILTER_MODEL_PATH:-/path/to/Qwen3-8B}"

# Hosted VL API (optional, for scripts/run_api.sh)
export PIGAI_API_APP_ID="${PIGAI_API_APP_ID:-}"
export PIGAI_API_APP_KEY="${PIGAI_API_APP_KEY:-}"
export PIGAI_API_ENDPOINT="${PIGAI_API_ENDPOINT:-https://your-api-endpoint/v1/chat/completions}"
