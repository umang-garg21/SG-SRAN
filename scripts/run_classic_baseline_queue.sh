#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 GPU_ID EXP_DIR [EXP_DIR ...]" >&2
    exit 2
fi

gpu_id="$1"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/data/home/umang/miniconda3/envs/material/bin/python"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLCONFIGDIR="/tmp/matplotlib"
export NUMBA_CACHE_DIR="/tmp/numba"

cd "${root}"
for exp_dir in "$@"; do
    echo "[$(date --iso-8601=seconds)] TRAIN ${exp_dir} on physical GPU ${gpu_id}"
    "${python_bin}" training/train_jangid_baseline.py \
        --exp_dir "${exp_dir}" \
        --resume \
        --skip_viz

    echo "[$(date --iso-8601=seconds)] INFER ${exp_dir}"
    "${python_bin}" inference/infer_jangid_baseline.py \
        --exp_dir "${exp_dir}" \
        --split Test \
        --max_visualizations 5
done

echo "[$(date --iso-8601=seconds)] QUEUE COMPLETE on physical GPU ${gpu_id}"
