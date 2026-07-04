#!/usr/bin/env bash
set -euo pipefail

interval="${1:-900}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

datasets=(IN718 Ti_Al_1pct)
methods=(rcan san han)
tasks=(4x1 4x4)

while true; do
    echo "=== $(date --iso-8601=seconds) ==="
    completed=0
    failed=0
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for task in "${tasks[@]}"; do
                exp="experiments/${dataset}/${method}_${task}_300ep_01"
                if [[ -f "${exp}/inference/test/summary.json" ]]; then
                    completed=$((completed + 1))
                    echo "COMPLETE ${dataset} ${method^^} ${task}"
                elif [[ -f "${exp}/logs/train.log" ]]; then
                    latest="$(rg '^Epoch ' "${exp}/logs/train.log" | tail -n 1 || true)"
                    echo "RUNNING  ${dataset} ${method^^} ${task}: ${latest:-initializing}"
                else
                    echo "QUEUED   ${dataset} ${method^^} ${task}"
                fi
            done
        done
    done
    shopt -s nullglob
    queue_logs=(
        experiments/_classic_baseline_queues/gpu[0-9].log
        experiments/_classic_baseline_queues/classic_g[0-9].log
    )
    for queue_log in "${queue_logs[@]}"; do
        if rg -q 'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|train=nan|val=nan' "${queue_log}"; then
            echo "FAILED QUEUE: ${queue_log}"
            tail -n 30 "${queue_log}"
            failed=1
        fi
    done
    shopt -u nullglob
    echo "Completed inference summaries: ${completed}/12"
    if [[ "${failed}" -ne 0 ]]; then
        exit 2
    fi
    if [[ "${completed}" -eq 12 ]]; then
        exit 0
    fi
    sleep "${interval}"
done
