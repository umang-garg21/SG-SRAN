#!/usr/bin/env bash
set -euo pipefail

# Run the Equiformer training with sensible defaults. Override via env vars
# or pass extra args which will be forwarded to the Python script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-material}"
LR_DIR="${LR_DIR:-/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/LR_Data}"
HR_DIR="${HR_DIR:-/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data}"
EPOCHS="${EPOCHS:-1000}"
VAL_EVERY="${VAL_EVERY:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PATIENCE="${PATIENCE:-5}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Try to initialize conda for scripts
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
fi

echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "Starting training:"
echo "  LR_DIR=$LR_DIR"
echo "  HR_DIR=$HR_DIR"
echo "  EPOCHS=$EPOCHS VAL_EVERY=$VAL_EVERY BATCH_SIZE=$BATCH_SIZE"
echo "  CKPT_DIR=$CKPT_DIR (timestamp subfolders will be created)"
echo "Logging to: $LOGFILE"

python train_equiformer.py \
  --lr_dir "$LR_DIR" \
  --hr_dir "$HR_DIR" \
  --epochs "$EPOCHS" \
  --val_every "$VAL_EVERY" \
  --batch_size "$BATCH_SIZE" \
  --ckpt_dir "$CKPT_DIR" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" "$@" 2>&1 | tee "$LOGFILE"

echo "Training finished. Logs written to $LOGFILE"
