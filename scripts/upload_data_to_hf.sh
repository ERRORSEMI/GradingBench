#!/usr/bin/env bash
# Upload data/ to Hugging Face dataset repo.
#
# Prereqs:
#   pip install -U huggingface_hub
#   hf auth login --token $HF_TOKEN    # write access to ERRORSEMI/GradingBench
#   export HF_ENDPOINT=https://hf-mirror.com   # if huggingface.co unreachable
#
# Before upload, sync dataset card for Hugging Face:
#   cp scripts/hf_dataset_card.md data/README.md
#
# Usage:
#   bash scripts/upload_data_to_hf.sh
#   HF_REPO=ERRORSEMI/GradingBench bash scripts/upload_data_to_hf.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
HF_REPO="${HF_REPO:-ERRORSEMI/GradingBench}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
COMMIT_MSG="${COMMIT_MSG:-Upload GradingBench data (images + annotations)}"
export HF_ENDPOINT

if [[ ! -d "$DATA_DIR/images" || ! -d "$DATA_DIR/annotations" ]]; then
  echo "Missing $DATA_DIR/images or annotations/"
  exit 1
fi

HF_BIN="${HF_BIN:-hf}"
if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "Install: pip install -U huggingface_hub"
  echo "Then login: hf auth login --token \$HF_TOKEN"
  exit 1
fi

if ! "$HF_BIN" auth whoami >/dev/null 2>&1; then
  echo "Not logged in. Run: hf auth login --token \$HF_TOKEN"
  exit 1
fi

echo "=== Upload to $HF_ENDPOINT (datasets/$HF_REPO) ==="
echo "logged in as: $("$HF_BIN" auth whoami 2>/dev/null | head -1 || true)"
echo "source=$DATA_DIR"
du -sh "$DATA_DIR" 2>/dev/null || true
echo

# New hf CLI: hf upload REPO LOCAL_PATH REMOTE_PATH --repo-type dataset
"$HF_BIN" upload "$HF_REPO" "$DATA_DIR" . \
  --repo-type dataset \
  --commit-message "$COMMIT_MSG"

echo "=== Done ==="
echo "https://huggingface.co/datasets/$HF_REPO"
