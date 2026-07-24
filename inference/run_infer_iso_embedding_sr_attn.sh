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
# Defaults
# ---------------------------------------------------------------------------
EXP_DIR="experiments/IN718/iso_embedding_sr_attn_01"  # relative to project root
CONFIG="config_new.json"          # auto-detect if not set
SPLIT="Test"
GPU_IDS="6"         # auto-detect if not set
CHECKPOINT="best_model.pt"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --exp_dir <path> [--config <file>] [--split Train|Val|Test] [--gpu <ids>] [--checkpoint <file>]"
    echo ""
    echo "  --exp_dir     Path to experiment directory (required). Can be absolute or"
    echo "                relative to the project root."
    echo "  --config      Config filename inside exp_dir (default: config_new.json if"
    echo "                present, otherwise config.json)."
    echo "  --split       Dataset split to run inference on (default: Test)."
    echo "  --gpu         CUDA_VISIBLE_DEVICES value, e.g. '0' or '4,5' (default: auto)."
    echo "  --checkpoint  Checkpoint filename in <exp_dir>/checkpoints/ (default: best_model.pt)."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_dir)    EXP_DIR="$2";    shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        --gpu)        GPU_IDS="$2";    shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$EXP_DIR" ]]; then
    echo "Error: --exp_dir is required."
    usage
fi

# Resolve exp_dir relative to project root if not absolute
if [[ "$EXP_DIR" != /* ]]; then
    EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
fi

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: $EXP_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect config
# ---------------------------------------------------------------------------
if [[ -z "$CONFIG" ]]; then
    if [[ -f "$EXP_DIR/config_new.json" ]]; then
        CONFIG="config_new.json"
    elif [[ -f "$EXP_DIR/config.json" ]]; then
        CONFIG="config.json"
    else
        echo "Error: no config_new.json or config.json found in $EXP_DIR"
        exit 1
    fi
fi

if [[ ! -f "$EXP_DIR/$CONFIG" ]]; then
    echo "Error: config file not found: $EXP_DIR/$CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify checkpoint exists
# ---------------------------------------------------------------------------
CKPT_PATH="$EXP_DIR/checkpoints/$CHECKPOINT"
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect a free GPU if not specified
# ---------------------------------------------------------------------------
if [[ -z "$GPU_IDS" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        # Pick the GPU with the most free memory
        GPU_IDS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t',' -k2 -rn \
            | head -1 \
            | awk -F',' '{print $1}' \
            | tr -d ' ')
        echo "Auto-selected GPU: $GPU_IDS"
    else
        GPU_IDS="0"
    fi
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"
mkdir -p "$EXP_DIR/logs"

echo "Experiment : $EXP_DIR"
echo "Config     : $CONFIG"
echo "Checkpoint : $CKPT_PATH"
echo "Split      : $SPLIT"
echo "GPU        : $GPU_IDS"
echo ""

CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m inference.infer_iso_embedding_sr_attn \
    --exp_dir "$EXP_DIR" \
    --config  "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split   "$SPLIT" \
    2>&1 | tee "$EXP_DIR/logs/infer_${SPLIT,,}.log"
