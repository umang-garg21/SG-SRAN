#!/usr/bin/env bash
# scripts/train_autoencoder.sh
# Launch FCC autoencoder training with proper environment setup.

set -e
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
    echo "   Usage: $0 <experiment_dir> [--config config.json] [--resume]"
    echo "   Example: $0 experiments/IN718/debug_x4 --config config_smoke.json"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Run training
# ---------------------------------------------------------------------------
echo "   Starting autoencoder training..."
echo "   Project root: $PROJECT_ROOT"
echo "   Experiment dir: $EXP_DIR"

python -m training.train_autoencoder --exp_dir "$EXP_DIR" $EXTRA_ARGS

echo "Autoencoder training completed for $EXP_DIR"
