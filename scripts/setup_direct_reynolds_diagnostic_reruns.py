#!/usr/bin/env python3
"""Prepare direct-Reynolds-isometric IN718 diagnostic reruns.

The current paper still labels the no-routing, no-residual, and HR-residual
sensitivity diagnostics as pre-refresh because those runs were produced with
the older Cartesian tensor-decomposition embedding. This helper clones the
current direct-Reynolds-isometric IN718 seed-42 config and changes only the
diagnostic knob under test.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/home/umang/miniconda3/envs/material/bin/python"
BASE_EXP = ROOT / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42"
OUT_ROOT = ROOT / "experiments/IN718/direct_reynolds_isometric_diagnostics"
LAUNCH_ROOT = ROOT / "experiments/direct_reynolds_isometric_diagnostics/_launch"


JOBS = [
    {
        "name": "no_routing_bicubic_s42",
        "gpu": 0,
        "model": {
            "type": "iso_embedding_no_routing",
            "model_module": "models.SR_4x4_no_routing",
            "model_class": "IsoEmbedding4x4NoRoutingBicubic",
        },
    },
    {
        "name": "no_routing_nn_s42",
        "gpu": 1,
        "model": {
            "type": "iso_embedding_no_routing",
            "model_module": "models.SR_4x4_no_routing",
            "model_class": "IsoEmbedding4x4NoRoutingNN",
        },
    },
    {
        "name": "no_residual_all_s42",
        "gpu": 2,
        "updates": {
            "use_residual_lr1": False,
            "use_residual_hr1": False,
            "use_residual_hr2": False,
            "use_residual_hr3": False,
            "lr_conv1_residual_weight": 0.0,
            "hr_conv1_residual_weight": 0.0,
            "hr_conv2_residual_weight": 0.0,
            "hr_conv3_residual_weight": 0.0,
        },
    },
    {
        "name": "hr_residual_w0p0_s42",
        "gpu": 3,
        "updates": {
            "use_residual_lr1": True,
            "use_residual_hr1": False,
            "use_residual_hr2": False,
            "use_residual_hr3": False,
            "hr_conv1_residual_weight": 0.0,
            "hr_conv2_residual_weight": 0.0,
            "hr_conv3_residual_weight": 0.0,
        },
    },
    {
        "name": "hr_residual_w0p1_s42",
        "gpu": 4,
        "updates": {
            "use_residual_hr1": True,
            "use_residual_hr2": True,
            "use_residual_hr3": True,
            "hr_conv1_residual_weight": 0.1,
            "hr_conv2_residual_weight": 0.1,
            "hr_conv3_residual_weight": 0.1,
        },
    },
    {
        "name": "hr_residual_w0p5_s42",
        "gpu": 5,
        "updates": {
            "use_residual_hr1": True,
            "use_residual_hr2": True,
            "use_residual_hr3": True,
            "hr_conv1_residual_weight": 0.5,
            "hr_conv2_residual_weight": 0.5,
            "hr_conv3_residual_weight": 0.5,
        },
    },
    {
        "name": "hr_residual_w1p0_s42",
        "gpu": 6,
        "updates": {
            "use_residual_hr1": True,
            "use_residual_hr2": True,
            "use_residual_hr3": True,
            "hr_conv1_residual_weight": 1.0,
            "hr_conv2_residual_weight": 1.0,
            "hr_conv3_residual_weight": 1.0,
        },
    },
]


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_config(base_cfg: dict, job: dict) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.pop("checkpoints_dir", None)
    cfg["seed"] = 42
    cfg["save_every"] = 0
    cfg["viz_every"] = 0
    cfg["final_viz"] = False
    cfg["plot_loss_curves"] = False
    cfg["save_last_checkpoint"] = False
    cfg["save_epoch_checkpoints"] = False
    cfg["print_model_summary"] = False
    logging_cfg = dict(cfg.get("logging", {}))
    logging_cfg["tensorboard"] = False
    logging_cfg["save_best_only"] = True
    cfg["logging"] = logging_cfg
    if "model" in job:
        cfg["model"] = job["model"]
    cfg.update(job.get("updates", {}))
    return cfg


def exp_dir(job: dict) -> Path:
    return OUT_ROOT / f"ocrp_direct_reynolds_isometric_l4_{job['name']}"


def write_run_script(job: dict) -> None:
    edir = exp_dir(job)
    rel = edir.relative_to(ROOT)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        'GPU_ID="${1:-0}"',
        f'EXP_DIR="{rel}"',
        'mkdir -p "$EXP_DIR/logs"',
        'export CUDA_VISIBLE_DEVICES="$GPU_ID"',
        f"export MPLCONFIGDIR=/tmp/matplotlib-direct-reynolds-diagnostics-{job['name']}",
        'export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"',
        'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
        'mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"',
        "",
        f"{PYTHON} training/train_iso_embedding_ocrp.py \\",
        '  --exp_dir "$EXP_DIR" \\',
        "  --config config_new.json \\",
        '  --gpu_ids "$GPU_ID"',
        "",
    ]
    path = edir / "run.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    make_executable(path)


def write_launchers() -> Path:
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    train_scripts = []
    infer_scripts = []
    statuses = []
    istatuses = []
    for job in JOBS:
        edir = exp_dir(job)
        rel = edir.relative_to(ROOT)
        gpu = int(job["gpu"])
        statuses.append(rel / "STATUS")
        istatuses.append(rel / "ISTATUS")

        train_path = LAUNCH_ROOT / f"train_{job['name']}_gpu{gpu}.sh"
        train_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -u",
                    f"cd {ROOT}",
                    f'if [ "$(cat {rel}/STATUS 2>/dev/null)" = DONE ]; then',
                    f'  echo "SKIP train {job["name"]}"',
                    "else",
                    f'  echo "[$(date +%F_%T)] TRAIN {job["name"]} gpu{gpu}"',
                    f'  bash {rel}/run.sh {gpu} > {rel}/logs/train.log 2>&1 '
                    f'&& echo DONE > {rel}/STATUS || echo FAILED > {rel}/STATUS',
                    "fi",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        make_executable(train_path)
        train_scripts.append(train_path.relative_to(ROOT))

        infer_path = LAUNCH_ROOT / f"infer_{job['name']}_gpu{gpu}.sh"
        out_dir = rel / "inference/test_best"
        infer_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -u",
                    f"cd {ROOT}",
                    f'if [ "$(cat {rel}/ISTATUS 2>/dev/null)" = IDONE ]; then',
                    f'  echo "SKIP infer {job["name"]}"',
                    "else",
                    f'  echo "[$(date +%F_%T)] INFER {job["name"]} gpu{gpu}"',
                    f"  {PYTHON} -m inference.infer_iso_embedding_sr_attn "
                    f"--exp_dir {rel} --config config_new.json --checkpoint best_model.pt "
                    f"--split Test --out_dir {out_dir} --gpu_ids {gpu} --skip_ipf "
                    f"> {rel}/logs/infer_test_best.log 2>&1 "
                    f"&& echo IDONE > {rel}/ISTATUS || echo IFAILED > {rel}/ISTATUS",
                    "fi",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        make_executable(infer_path)
        infer_scripts.append(infer_path.relative_to(ROOT))

    master = LAUNCH_ROOT / "run_train_infer.sh"
    master.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {ROOT}",
                "TRAIN_JOBS=(" + " ".join(str(p) for p in train_scripts) + ")",
                "INFER_JOBS=(" + " ".join(str(p) for p in infer_scripts) + ")",
                "TRAIN_STATUS=(" + " ".join(str(p) for p in statuses) + ")",
                "INFER_STATUS=(" + " ".join(str(p) for p in istatuses) + ")",
                "",
                "echo '[diagnostics] training direct-Reynolds diagnostic reruns'",
                "pids=()",
                'for job in "${TRAIN_JOBS[@]}"; do bash "$job" > "$job.log" 2>&1 & pids+=("$!"); done',
                'for pid in "${pids[@]}"; do wait "$pid"; done',
                'for st in "${TRAIN_STATUS[@]}"; do',
                '  if [ "$(cat "$st" 2>/dev/null)" != DONE ]; then echo "[diagnostics] failed train status: $st"; exit 1; fi',
                "done",
                "",
                "echo '[diagnostics] running test inference from best checkpoints'",
                "pids=()",
                'for job in "${INFER_JOBS[@]}"; do bash "$job" > "$job.log" 2>&1 & pids+=("$!"); done',
                'for pid in "${pids[@]}"; do wait "$pid"; done',
                'for st in "${INFER_STATUS[@]}"; do',
                '  if [ "$(cat "$st" 2>/dev/null)" != IDONE ]; then echo "[diagnostics] failed infer status: $st"; exit 1; fi',
                "done",
                "",
                "echo '[diagnostics] done'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    make_executable(master)
    return master


def main() -> None:
    base_cfg = json.loads((BASE_EXP / "config_new.json").read_text(encoding="utf-8"))
    for job in JOBS:
        edir = exp_dir(job)
        (edir / "logs").mkdir(parents=True, exist_ok=True)
        (edir / "checkpoints").mkdir(exist_ok=True)
        write_json(edir / "config_new.json", make_config(base_cfg, job))
        write_run_script(job)
        status = edir / "STATUS"
        if not status.exists():
            status.write_text("PENDING\n", encoding="utf-8")
    master = write_launchers()
    print(f"Prepared {len(JOBS)} diagnostic jobs under {OUT_ROOT.relative_to(ROOT)}")
    print(f"Master script: {master.relative_to(ROOT)}")
    for job in JOBS:
        print(f"  gpu{job['gpu']}: {exp_dir(job).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
