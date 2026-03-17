#!/usr/bin/env bash
# Run inference for IsoEmbeddingSRAttn.

set -e
set -o pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

EXP_DIR="$1"
shift || true
EXTRA_ARGS="$@"

if [ -z "$EXP_DIR" ]; then
    echo "Usage: $0 <experiment_dir> [--checkpoint best_model.pt] [--split Test] [--take_first 8]"
    echo "Example: $0 experiments/IN718/iso_embedding_sr_attn_01 --checkpoint best_model.pt --split Test"
    exit 1
fi

echo "Starting IsoEmbeddingSRAttn inference..."
echo "Project root: $PROJECT_ROOT"
echo "Experiment dir: $EXP_DIR"

python -m inference.infer_iso_embedding_sr_attn --exp_dir "$EXP_DIR" $EXTRA_ARGS

echo "Inference completed for $EXP_DIR"
