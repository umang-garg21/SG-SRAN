#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_DIR="$PROJECT_ROOT/experiments/IN718/irrep_sdf_sr_one_sided_modified_compact_norefine_01"
CONFIG="config_new.json"
GPU_IDS="0"
RESUME=""
CHECK_WARNINGS="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="--resume"; shift ;;
        --gpu) GPU_IDS="$2"; shift 2 ;;
        --exp_dir) EXP_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --no_warning_check) CHECK_WARNINGS="0"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"
mkdir -p "$EXP_DIR/logs"

LOG_PATH="$EXP_DIR/logs/train.log"

python -m training.train_iso_embedding_sr_attn \
    --exp_dir "$EXP_DIR" \
    --config "$CONFIG" \
    --gpu_ids "$GPU_IDS" \
    $RESUME \
    2>&1 | tee "$LOG_PATH"

if [[ "$CHECK_WARNINGS" == "1" && -f "$LOG_PATH" ]]; then
    if rg -n "unexpected keyword argument 'lr_boundary_map'|Failed to render LR/SR/HR IPF visualization" "$LOG_PATH" >/dev/null; then
        echo "[warning-check] Found boundary/IPF warning patterns in $LOG_PATH"
        rg -n "unexpected keyword argument 'lr_boundary_map'|Failed to render LR/SR/HR IPF visualization" "$LOG_PATH" || true
    else
        echo "[warning-check] No boundary/IPF warning patterns found in $LOG_PATH"
    fi
fi
