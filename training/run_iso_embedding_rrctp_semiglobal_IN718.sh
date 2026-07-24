#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CALLER_PWD="$PWD"

EXP_DIR="$PROJECT_ROOT/experiments/IN718/iso_embedding_rrctp_semiglobal_01"
CONFIG="config_new.json"
GPU_IDS="0"
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="--resume"; shift ;;
        --gpu) GPU_IDS="$2"; shift 2 ;;
        --exp_dir) EXP_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$EXP_DIR" != /* ]]; then
    CAND_CALLER="$CALLER_PWD/$EXP_DIR"
    CAND_PROJECT="$PROJECT_ROOT/$EXP_DIR"
    if [[ -d "$CAND_CALLER" || -f "$CAND_CALLER/$CONFIG" ]]; then
        EXP_DIR="$CAND_CALLER"
    elif [[ -d "$CAND_PROJECT" || -f "$CAND_PROJECT/$CONFIG" ]]; then
        EXP_DIR="$CAND_PROJECT"
    else
        EXP_DIR="$CAND_CALLER"
    fi
fi
EXP_DIR="$(realpath -m "$EXP_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -f "$EXP_DIR/$CONFIG" ]]; then
    echo "Error: config file not found: $EXP_DIR/$CONFIG"
    echo "Tip: pass --exp_dir as the experiment directory containing $CONFIG"
    exit 1
fi

mkdir -p "$EXP_DIR/logs"

python -m training.train_iso_embedding_sr_attn \
    --exp_dir "$EXP_DIR" \
    --config "$CONFIG" \
    --gpu_ids "$GPU_IDS" \
    $RESUME \
    2>&1 | tee "$EXP_DIR/logs/train.log"
