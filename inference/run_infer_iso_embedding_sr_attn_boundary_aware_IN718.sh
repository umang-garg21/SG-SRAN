#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate material

# ---------------------------------------------------------------------------
# Defaults for boundary-aware IN718 experiment
# ---------------------------------------------------------------------------
EXP_DIR="$PROJECT_ROOT/experiments/IN718/iso_embedding_sr_attn_boundary_aware_01"
CONFIG="config_new.json"
SPLIT="Test"
GPU_IDS=""
CHECKPOINT="best_model.pt"
MAX_BATCHES=""
TAKE_FIRST=""
OUT_DIR=""
VIZ_REF_DIR="ALL"
ALLOW_OTHER_MODEL=0

usage() {
    echo "Usage: $0 [--exp_dir <path>] [--config <file>] [--checkpoint <file>] [--split Train|Val|Test]"
    echo "          [--gpu <ids>] [--max_batches <n>] [--take_first <n>] [--out_dir <path>]"
    echo "          [--viz_ref_dir X|Y|Z|ALL] [--allow_other_model]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_dir)           EXP_DIR="$2"; shift 2 ;;
        --config)            CONFIG="$2"; shift 2 ;;
        --checkpoint)        CHECKPOINT="$2"; shift 2 ;;
        --split)             SPLIT="$2"; shift 2 ;;
        --gpu)               GPU_IDS="$2"; shift 2 ;;
        --max_batches)       MAX_BATCHES="$2"; shift 2 ;;
        --take_first)        TAKE_FIRST="$2"; shift 2 ;;
        --out_dir)           OUT_DIR="$2"; shift 2 ;;
        --viz_ref_dir)       VIZ_REF_DIR="$2"; shift 2 ;;
        --allow_other_model) ALLOW_OTHER_MODEL=1; shift ;;
        -h|--help)           usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ "$EXP_DIR" != /* ]]; then
    EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
fi
if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: $EXP_DIR"
    exit 1
fi
if [[ ! -f "$EXP_DIR/$CONFIG" ]]; then
    echo "Error: config file not found: $EXP_DIR/$CONFIG"
    exit 1
fi
if [[ "$CHECKPOINT" == /* ]]; then
    CKPT_PATH="$CHECKPOINT"
else
    CKPT_PATH="$EXP_DIR/checkpoints/$CHECKPOINT"
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Auto-select GPU by most free memory if not explicitly provided.
if [[ -z "$GPU_IDS" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_IDS="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t',' -k2 -rn \
            | head -1 \
            | awk -F',' '{print $1}' \
            | tr -d ' ')"
    else
        GPU_IDS="0"
    fi
fi

mkdir -p "$EXP_DIR/logs"
LOG_PATH="$EXP_DIR/logs/infer_boundary_aware_${SPLIT,,}.log"

echo "Experiment : $EXP_DIR"
echo "Config     : $CONFIG"
echo "Checkpoint : $CKPT_PATH"
echo "Split      : $SPLIT"
echo "GPU        : $GPU_IDS"
echo "Log        : $LOG_PATH"
echo ""

CMD=(python -m inference.infer_iso_embedding_sr_attn_boundary_aware
    --exp_dir "$EXP_DIR"
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
    --viz_ref_dir "$VIZ_REF_DIR"
)

if [[ -n "$MAX_BATCHES" ]]; then
    CMD+=(--max_batches "$MAX_BATCHES")
fi
if [[ -n "$TAKE_FIRST" ]]; then
    CMD+=(--take_first "$TAKE_FIRST")
fi
if [[ -n "$OUT_DIR" ]]; then
    CMD+=(--out_dir "$OUT_DIR")
fi
if [[ "$ALLOW_OTHER_MODEL" -eq 1 ]]; then
    CMD+=(--allow_other_model)
fi

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}" 2>&1 | tee "$LOG_PATH"

