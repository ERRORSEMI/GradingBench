#!/usr/bin/env bash
# Fast bulk copy: one cp/rsync per (level, subject) directory (~15 ops, not 5000+).
# Usage: bash scripts/copy_images.sh /path/to/legacy/test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${1:?usage: bash scripts/copy_images.sh /path/to/legacy/test}"
DEST="${REPO}/data/images"

declare -A SUBJ=(
  [数学]=Mathematics
  [语文]=Chinese
  [英语]=English
  [理综]=Science
  [文综]=Humanities
)

copy_level() {
  local level="$1" src_subdir="$2"
  local cn en src_dir dst_dir
  echo "=== ${level} (${src_subdir}) ==="
  for cn in "${!SUBJ[@]}"; do
    en="${SUBJ[$cn]}"
    src_dir="${SRC}/${cn}/${src_subdir}"
    dst_dir="${DEST}/${level}/${en}"
    mkdir -p "${dst_dir}"
    if [[ ! -d "${src_dir}" ]]; then
      echo "  skip ${cn}: no ${src_dir}"
      continue
    fi
    n=$(find "${src_dir}" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
    echo -n "  ${en}: ${n} files ... "
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --info=stats2 "${src_dir}/" "${dst_dir}/" 2>&1 | tail -1 || true
    else
      cp -f "${src_dir}"/*.jpg "${dst_dir}/" 2>/dev/null || true
      echo "cp done"
    fi
  done
}

copy_level L1 "L1-单题裁剪图"
copy_level L2 "L2-整页原图"
copy_level L3 "L3-整页原图"

total=$(find "${DEST}" -name '*.jpg' 2>/dev/null | wc -l)
echo ""
echo "Done: ${total} images under ${DEST}/"
