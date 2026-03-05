#!/usr/bin/env bash
# scripts/train.sh
# Launch Reynolds-QSR training run with proper environment setup.

set -e  # exit on first error
set -o pipefail

# ---------------------------------------------------------------------------
# 1. Resolve project root and set PYTHONPATH
# ---------------------------------------------------------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 2. Parse optional args
# ---------------------------------------------------------------------------
EXP_DIR="$1"
shift || true
EXTRA_ARGS="$@"

if [ -z "$EXP_DIR" ]; then
    echo "   Usage: $0 <experiment_dir> [--resume]"
    echo "   Example: $0 experiments/IN718/debug_x4 --resume"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Run training
# ---------------------------------------------------------------------------
echo "   Starting training..."
echo "   Project root: $PROJECT_ROOT"
echo "   Experiment dir: $EXP_DIR"

python -m training.train_sr --exp_dir "$EXP_DIR" $EXTRA_ARGS

echo "Training completed for $EXP_DIR"
