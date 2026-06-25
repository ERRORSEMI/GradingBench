#!/usr/bin/env bash
# shell 脚本公共路径初始化

pigai_init_shell() {
  local level="${1:?usage: pigai_init_shell L1|L2|L3}"
  local script_path="${BASH_SOURCE[1]:-$0}"
  local script_dir
  script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
  export PIGAI_SHELL_DIR="${script_dir}"
  export PIGAI_WORKPY_ROOT="$(cd "${script_dir}/../.." && pwd)"
  export PIGAI_REPO_ROOT="$(cd "${PIGAI_WORKPY_ROOT}/.." && pwd)"
  export PIGAI_LEVEL="${level}"
  export PIGAI_LEVEL_DIR="${PIGAI_WORKPY_ROOT}/${level}-pigaipy"
  # shellcheck source=/dev/null
  source "${PIGAI_REPO_ROOT}/scripts/env.sh"
}
