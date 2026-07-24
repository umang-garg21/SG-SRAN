#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 GPU_ID EPOCH EXP_DIR [EXP_DIR ...]" >&2
    exit 2
fi

gpu_id="$1"
threshold="$2"
shift 2
experiments=("$@")
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/data/home/umang/miniconda3/envs/material/bin/python"
checkpoint_name="epoch_$(printf '%04d' "${threshold}").pt"
output_name="test_epoch$(printf '%04d' "${threshold}")"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLCONFIGDIR="/tmp/matplotlib"
export NUMBA_CACHE_DIR="/tmp/numba"
cd "${root}"

remaining=("${experiments[@]}")
while [[ ${#remaining[@]} -gt 0 ]]; do
    next=()
    for exp_dir in "${remaining[@]}"; do
        checkpoint="${exp_dir}/checkpoints/${checkpoint_name}"
        summary="${exp_dir}/inference/${output_name}/summary.json"
        if [[ -f "${summary}" ]]; then
            echo "Already complete: ${summary}"
            continue
        fi
        if [[ ! -f "${checkpoint}" ]]; then
            next+=("${exp_dir}")
            continue
        fi

        # A checkpoint pathname appears while torch.save is writing.  Require a
        # successful load before inference so training and inference cannot race.
        if ! "${python_bin}" -c "import torch; torch.load('${checkpoint}', map_location='cpu')" >/dev/null 2>&1; then
            next+=("${exp_dir}")
            continue
        fi

        echo "[$(date --iso-8601=seconds)] INFER ${exp_dir} at epoch ${threshold}"
        "${python_bin}" inference/infer_jangid_baseline.py \
            --exp_dir "${exp_dir}" \
            --checkpoint "${checkpoint_name}" \
            --split Test \
            --out_dir "${exp_dir}/inference/${output_name}" \
            --max_visualizations 5 \
            > "${exp_dir}/logs/infer_epoch$(printf '%04d' "${threshold}").log" 2>&1
    done
    remaining=("${next[@]}")
    if [[ ${#remaining[@]} -gt 0 ]]; then
        sleep 60
    fi
done

echo "[$(date --iso-8601=seconds)] THRESHOLD INFERENCE QUEUE COMPLETE"
