#!/usr/bin/env bash
# 删除旧版 data 目录结构
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data"

echo "Cleaning ${ROOT} ..."

rm -rf \
  "${ROOT}/L1-Single-Question" \
  "${ROOT}/L2-Specified-Question" \
  "${ROOT}/L3-Full-Page" \
  "${ROOT}/annotations/L1" \
  "${ROOT}/annotations/L2" \
  "${ROOT}/annotations/L3" \
  "${ROOT}/test"

rm -f "${ROOT}/manifest/L1.json" \
      "${ROOT}/manifest/L2.json" \
      "${ROOT}/manifest/L3.json" \
      "${ROOT}/manifest/L1-Single-Question.json" \
      "${ROOT}/manifest/L2-Specified-Question.json" \
      "${ROOT}/manifest/L3-Full-Page.json"

# 旧版 per-subject jsonl
find "${ROOT}" -name 'annotations.jsonl' -delete 2>/dev/null || true

echo "Target layout: images/L{1,2,3}/{Subject}/ + annotations/L{1,2,3}.json"
