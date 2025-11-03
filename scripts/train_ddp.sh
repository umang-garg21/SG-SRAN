#!/bin/bash
# Launch script for Distributed Data Parallel (DDP) training with dynamic batch sizing

# Usage:
#   ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/ [--resume]
#
# Examples:
#   # Start new training
#   ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/
#
#   # Resume training
#   ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/ --resume

set -e

# Get experiment directory
EXP_DIR="${1:?Please provide experiment directory as first argument}"
RESUME_FLAG=""

# Check for resume flag
if [ "$2" == "--resume" ]; then
    RESUME_FLAG="--resume"
fi

echo "=========================================="
echo "  DDP Training Configuration"
echo "=========================================="
echo "Experiment dir: $EXP_DIR"
echo "Max GPUs: 6 (will use available up to this limit)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all available}"
echo "Resume: ${RESUME_FLAG:-false}"
echo "Dynamic batch size: Enabled"
echo "  - Initial batch size from config"
echo "  - Constant until epoch 500"
echo "  - Then divide by 2 every 100 epochs"
echo "  - After reaching 4: Alternate 4/1 every 100 epochs"
echo
echo "=========================================="

# Get project root (assuming script is in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "   Starting DDP training..."
echo "   Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run the DDP training script (uses mp.spawn internally)
python -m training.train_sr_ddp \
    --exp_dir "$EXP_DIR" \
    $RESUME_FLAG

echo
echo "=========================================="
echo "DDP training complete!"
echo "=========================================="
