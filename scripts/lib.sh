#!/usr/bin/env bash
# shell 脚本公共路径初始化

pigai_init_shell() {
  local level="${1:?usage: pigai_init_shell L1|L2|L3}"
  local script_path="${BASH_SOURCE[1]:-$0}"
  local script_dir
  script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
  export PIGAI_SHELL_DIR="${script_dir}"
  export PIGAI_REPO_ROOT="$(cd "${script_dir}/.." && pwd)"
  export PIGAI_SRC_ROOT="${PIGAI_REPO_ROOT}/src"
  export PIGAI_LEVEL="${level}"
  export PYTHONPATH="${PIGAI_SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  # shellcheck source=/dev/null
  source "${PIGAI_REPO_ROOT}/scripts/env.sh"
}
