#!/usr/bin/env python3
"""批量脱敏开源仓库：替换绝对路径、删除密钥、统一 shell 入口。"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    (re.compile(r'PIGAI_API_APP_ID\s*=\s*["\'][0-9]+["\']'), 'PIGAI_API_APP_ID = os.environ.get("PIGAI_API_APP_ID", "")'),
    (re.compile(r'PIGAI_API_APP_KEY\s*=\s*["\'][0-9a-f]+["\']'), 'PIGAI_API_APP_KEY = os.environ.get("PIGAI_API_APP_KEY", "")'),
    (re.compile(r'TAL_MLOPS_APP_ID\s*=\s*["\'][0-9]+["\']'), 'PIGAI_API_APP_ID = os.environ.get("PIGAI_API_APP_ID", "")'),
    (re.compile(r'TAL_MLOPS_APP_KEY\s*=\s*["\'][0-9a-f]+["\']'), 'PIGAI_API_APP_KEY = os.environ.get("PIGAI_API_APP_KEY", "")'),
]

PATH_REPLACEMENTS = [
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令/", "${PIGAI_REPO_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令开源/", "${PIGAI_REPO_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/data/test", "${PIGAI_DATA_ROOT}"),
    ("${LEGACY_ROOT}/model/", "${MODEL_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/微调/", "${FINETUNE_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/1-定位/", "${PIGAI_LOC1_ROOT}/"),
    ("${LEGACY_ROOT}/miniconda3/", "${CONDA_ROOT}/"),
    ("/mnt/pfs_l2/jieti_team/CV/wangyuting/", "${USER_HOME}/"),
    ("/mnt/pfs_l2/jieti_team/MMGroup/miniconda3/", "${CONDA_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令/workpy/", "${PIGAI_WORKPY_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令开源/workpy/", "${PIGAI_WORKPY_ROOT}/"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令/结果", "${PIGAI_RESULTS_ROOT}"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令开源/结果", "${PIGAI_RESULTS_ROOT}"),
    ("${LEGACY_ROOT}/pigaiwork/0-综合复杂指令/workpy/common/fonts/simhei.ttf", "${PIGAI_FONT_PATH}"),
]

EVALUATOR_TEMPLATE = '''#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)/../scripts/lib.sh"
pigai_init_shell "{level}"

DEFAULT_MODEL="${{DEFAULT_MODEL:-glm-4.6v}}"
MODEL="$DEFAULT_MODEL"
NEED_ANSWER="False"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --need_answer) NEED_ANSWER="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "=== [{level}] model=$MODEL need_answer=$NEED_ANSWER ==="

python "${{PIGAI_LEVEL_DIR}}/stage1_generate_raw_results.py" \\
    --model "$MODEL" --need_answer "$NEED_ANSWER"

python "${{PIGAI_LEVEL_DIR}}/stage2_filter_with_qwen.py" \\
    --tensor_parallel_size "${{TENSOR_PARALLEL_SIZE:-2}}" \\
    --model "$MODEL" --need_answer "$NEED_ANSWER" \\
    --model_path "${{PIGAI_FILTER_MODEL_PATH}}"

python "${{PIGAI_LEVEL_DIR}}/stage3_calculate_metrics.py" \\
    --model "$MODEL" --need_answer "$NEED_ANSWER"

echo "=== Done ==="
'''


def sanitize_text(text: str, is_shell: bool) -> str:
    # 删除明文 key
    text = re.sub(r'(?m)^\s*#?\s*(TAL_MLOPS_APP_ID|PIGAI_API_APP_ID)\s*=.*$\n?', "", text)
    text = re.sub(r'(?m)^\s*#?\s*(TAL_MLOPS_APP_KEY|PIGAI_API_APP_KEY)\s*=.*$\n?', "", text)
    text = re.sub(r'435d6186781dce102605e53357f6f25b', "<REDACTED>", text)
    text = re.sub(r'3d3b86c1aae87f930a3b92b5cab46543', "<REDACTED>", text)
    text = re.sub(r'ff0942f600160a2cf11c37a60f215c20', "<REDACTED>", text)
    text = re.sub(
        r'https?://[^\s"\']+/(openai-compatible/)?v1/chat/completions',
        "${PIGAI_API_ENDPOINT}",
        text,
    )

    for old, new in PATH_REPLACEMENTS:
        text = text.replace(old, new)

    if is_shell:
        # 去掉硬编码 conda activate
        text = re.sub(
            r'^source\s+\$\{CONDA_ROOT\}.*\n',
            '# 请先: conda activate your-env\n',
            text,
            flags=re.M,
        )
        text = re.sub(
            r'^source\s+/mnt/pfs_l2/.*conda.*\n',
            '# 请先: conda activate your-env\n',
            text,
            flags=re.M,
        )
    return text


def patch_stage2(path: Path):
    text = path.read_text(encoding="utf-8")
    if "FILTER_MODEL_PATH" in text:
        return
    insert = '''import os
import sys
_WORKPY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKPY not in sys.path:
    sys.path.insert(0, os.path.dirname(_WORKPY))
from paths import FILTER_MODEL_PATH
'''
    if "from config import create_config" in text and "from paths import" not in text:
        text = text.replace("import os\n", insert, 1)
    text = re.sub(
        r'default="/mnt/pfs_l2[^"]+"',
        'default=FILTER_MODEL_PATH',
        text,
    )
    text = re.sub(
        r'default="\$\{MODEL_ROOT\}/qwen/Qwen3-8B"',
        'default=FILTER_MODEL_PATH',
        text,
    )
    path.write_text(text, encoding="utf-8")


def main():
    skip = {"prepare_opensource_release.py", "sanitize_repo.py", "paths.py"}
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".sh"}:
            continue
        if path.name in skip:
            continue
        if "egg-info" in str(path):
            continue
        if path.name in {"config.py", "model_init.py"} and "config" in str(path) or path.name == "model_init.py":
            continue  # already hand-edited

        raw = path.read_text(encoding="utf-8", errors="ignore")
        new = sanitize_text(raw, path.suffix == ".sh")
        if new != raw:
            path.write_text(new, encoding="utf-8")
            print(f"patched: {path.relative_to(REPO)}")

    for level in ("L1", "L2", "L3"):
        ev = REPO / "workpy" / f"{level}-pigaipy" / "shells" / "evaluator.sh"
        ev.write_text(EVALUATOR_TEMPLATE.format(level=level), encoding="utf-8")
        ev.chmod(0o755)
        print(f"rewrote: {ev.relative_to(REPO)}")

        s2 = REPO / "workpy" / f"{level}-pigaipy" / "stage2_filter_with_qwen.py"
        if s2.exists():
            patch_stage2(s2)

    print("\nDone.")


if __name__ == "__main__":
    main()
