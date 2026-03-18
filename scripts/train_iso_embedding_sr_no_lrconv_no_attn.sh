#!/usr/bin/env bash
# Launch no-LR-conv/no-attention ablation training.

set -e
set -o pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

EXP_DIR="${1:-experiments/Ti64/iso_embedding_sr_no_lrconv_no_attn_hcp_01}"
shift || true
EXTRA_ARGS="$@"

echo "Starting IsoEmbeddingSRAttn ablation training (no LR conv1/2, no attention)..."
echo "Project root: $PROJECT_ROOT"
echo "Experiment dir: $EXP_DIR"

python -B -m training.train_iso_embedding_sr_attn \
  --exp_dir "$EXP_DIR" \
  --config config.json \
  $EXTRA_ARGS

echo "Training completed for $EXP_DIR"
