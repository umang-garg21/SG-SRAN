#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_path="$repo_root/.cache/jupyter-kernels/share/jupyter"

export JUPYTER_PATH="$kernel_path${JUPYTER_PATH:+:$JUPYTER_PATH}"

if [[ $# -eq 0 ]]; then
  exec jupyter lab
fi

exec jupyter "$@"
