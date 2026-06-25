#!/usr/bin/env bash
# Export release dataset: images/ + annotations/L{1,2,3}.json
# Usage: bash scripts/prepare_release_data.sh /path/to/legacy/test [annotations-only|link|copy-all]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:?usage: prepare_release_data.sh /path/to/legacy/test [mode]}"
MODE="${2:-link}"

bash "${SCRIPT_DIR}/cleanup_legacy_data.sh"
python3 "${SCRIPT_DIR}/prepare_dataset_release.py" --source "$SRC" --mode "$MODE"
echo "Done: data/images/ + data/annotations/L1.json L2.json L3.json"
