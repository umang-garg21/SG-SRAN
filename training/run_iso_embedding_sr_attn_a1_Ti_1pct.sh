#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_DIR="$PROJECT_ROOT/experiments/Ti_Al_1pct/iso_embedding_sr_attn_hcp_LR_attn_HR_attn_01"
CONFIG="config_new.json"
GPU_IDS="0"
RESUME=""

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="--resume"; shift ;;
        --gpu) GPU_IDS="$2"; shift 2 ;;
        --exp_dir) EXP_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

mkdir -p "$EXP_DIR/logs"

python -m training.train_iso_embedding_sr_attn \
    --exp_dir "$EXP_DIR" \
    --config "$CONFIG" \
    --gpu_ids "$GPU_IDS" \
    $RESUME \
    2>&1 | tee "$EXP_DIR/logs/train.log"
