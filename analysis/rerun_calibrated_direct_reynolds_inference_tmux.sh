#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home/umang/Materials/Reynolds-QSR_paper"
PY="/data/home/umang/miniconda3/envs/material/bin/python"
SEEDS=(42 43 44 45 46)
GPUS=(0 1 2 3 4)
RUN_ID="${RUN_ID:-calibrated_inference_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
mkdir -p analysis/out

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

backup_inference_dir() {
  local exp_dir="$1"
  local out_dir="$exp_dir/inference/test_best"
  if [ -e "$out_dir" ]; then
    local backup_dir="$exp_dir/inference/test_best_pre_${RUN_ID}"
    if [ -e "$backup_dir" ]; then
      backup_dir="${backup_dir}_$RANDOM"
    fi
    mv "$out_dir" "$backup_dir"
  fi
  mkdir -p "$exp_dir/inference"
}

run_seed_inference() {
  local material="$1"
  local prefix="$2"
  local seed="$3"
  local gpu="$4"
  local exp_root="$5"
  local exp_dir="$exp_root/${prefix}_s${seed}"
  local log_dir="$exp_dir/logs"
  local infer_log="$log_dir/infer_test_best_${RUN_ID}.log"

  mkdir -p "$log_dir"
  backup_inference_dir "$exp_dir"
  echo "IRUNNING_${RUN_ID}" > "$exp_dir/ISTATUS"
  log "$material seed $seed inference on GPU $gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" inference/infer_iso_embedding_sr_attn.py \
      --exp_dir "$exp_dir" \
      --config config_new.json \
      --checkpoint best_model.pt \
      --split Test \
      --out_dir "$exp_dir/inference/test_best" \
      --skip_ipf \
      --gpu_ids "$gpu" > "$infer_log" 2>&1; then
    echo "IDONE_${RUN_ID}" > "$exp_dir/ISTATUS"
    log "$material seed $seed inference done"
  else
    echo "IFAILED_${RUN_ID}" > "$exp_dir/ISTATUS"
    log "$material seed $seed inference failed; see $infer_log"
    return 1
  fi
}

run_material() {
  local material="$1"
  local exp_root="$2"
  local prefix="$3"
  local metrics_gpu="$4"
  local pids=()
  local seed gpu idx fail=0

  log "Starting $material calibrated inference rerun: ${SEEDS[*]}"
  for idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[$idx]}"
    gpu="${GPUS[$idx]}"
    run_seed_inference "$material" "$prefix" "$seed" "$gpu" "$exp_root" &
    pids+=("$!")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      fail=1
    fi
  done

  if [ "$fail" -ne 0 ]; then
    log "$material had at least one failed inference; skipping material metrics"
    return 1
  fi

  log "$material inference complete; collecting metrics"
  CUDA_VISIBLE_DEVICES="$metrics_gpu" "$PY" analysis/collect_direct_reynolds_isometric_material_metrics.py \
    --material "$material" \
    --out-prefix "analysis/out/${material}_direct_reynolds_isometric_${RUN_ID}" \
    --seeds "${SEEDS[@]}" \
    > "analysis/out/${material}_direct_reynolds_isometric_${RUN_ID}_metrics.log" 2>&1
  log "$material metrics done"
}

log "RUN_ID=$RUN_ID"
run_material "IN718" "experiments/IN718/direct_reynolds_isometric_seed_runs" "ocrp_direct_reynolds_isometric_l4" 0
run_material "Ti_Al_1pct" "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs" "ocrp_direct_reynolds_isometric_l6" 0
log "All calibrated inference reruns complete"
