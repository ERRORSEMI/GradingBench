#!/usr/bin/env bash
# Download GradingBench data from Hugging Face into ./data/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

HF_REPO="${HF_DATASET_REPO:-ERRORSEMI/GradingBench}"
DEST="${PIGAI_DATA_ROOT:-${REPO}/data}"
HF_BIN="${HF_BIN:-hf}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "Install: pip install -U huggingface_hub"
  echo "Then login (if needed): hf auth login"
  exit 1
fi

echo "=== Download dataset ==="
echo "repo:  $HF_REPO (datasets)"
echo "dest:  $DEST"
echo "url:   https://huggingface.co/datasets/$HF_REPO"
echo

mkdir -p "$DEST"
"$HF_BIN" download "$HF_REPO" --repo-type dataset --local-dir "$DEST"

echo
echo "Done. Expected layout:"
echo "  $DEST/images/L{1,2,3}/{Subject}/*.jpg"
echo "  $DEST/annotations/L{1,2,3}.json"
