#!/usr/bin/env bash
# Launch IsoEmbeddingSRAttn training.

set -e
set -o pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

EXP_DIR="$1"
shift || true
EXTRA_ARGS="$@"

if [ -z "$EXP_DIR" ]; then
    echo "Usage: $0 <experiment_dir> [--config config_smoke.json] [--resume] [--gpu_ids 0]"
    echo "Example: $0 experiments/IN718/iso_embedding_sr_attn_01 --config config_smoke.json"
    exit 1
fi

echo "Starting IsoEmbeddingSRAttn training..."
echo "Project root: $PROJECT_ROOT"
echo "Experiment dir: $EXP_DIR"

python -m training.train_iso_embedding_sr_attn --exp_dir "$EXP_DIR" $EXTRA_ARGS

echo "Training completed for $EXP_DIR"
