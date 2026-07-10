#!/usr/bin/env python3
"""Prepare corrected direct-Reynolds OCRP 2x2/8x8 scaling reruns.

The old scaling panel points at legacy tensor-product OCRP runs and at dataset
roots that are not mounted in this workspace.  This script rebuilds the missing
2x2 and 8x8 datasets from the current x4 HR tiles, creates seed-42
direct-Reynolds-isometric OCRP experiment configs, and writes a master launcher
that trains and evaluates the four scale/material jobs in parallel.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/home/umang/miniconda3/envs/material/bin/python"
RUN_ROOT = ROOT / "experiments/direct_reynolds_isometric_scaling"
DATA_ROOT = RUN_ROOT / "datasets"
LAUNCH_ROOT = RUN_ROOT / "_launch"

SCALES = (2, 8)

MATERIALS = {
    "IN718": {
        "source_dataset": Path("/data/home/umang/Materials/Materials_data_mount/datasets/IN718_QSR_x4"),
        "source_run": ROOT
        / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42",
        "legacy_scale_runs": {
            2: ROOT / "experiments/IN718/iso_embedding_2x2_ocrp_01",
            8: ROOT / "experiments/IN718/iso_embedding_8x8_ocrp_01",
        },
        "symmetry": "Oh",
        "crystal": "fcc",
        "harmonic_l": 4,
        "prefix": "ocrp_direct_reynolds_isometric_l4",
    },
    "Ti_Al_1pct": {
        "source_dataset": Path("/data/home/umang/Materials/Materials_data_mount/datasets/Ti_Al_1pct_QSR_x4"),
        "source_run": ROOT
        / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42",
        "legacy_scale_runs": {
            2: ROOT / "experiments/Ti_Al_1pct/iso_embedding_2x2_ocrp_01",
            8: ROOT / "experiments/Ti_Al_1pct/iso_embedding_8x8_ocrp_01",
        },
        "symmetry": "D6h",
        "crystal": "hcp",
        "harmonic_l": 6,
        "prefix": "ocrp_direct_reynolds_isometric_l6",
    },
}

NAME_RE = re.compile(
    r"^(?P<ds>.+)_x4_(?P<split>train|val|test)_hr_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
    re.IGNORECASE,
)


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dataset_name(material: str, scale: int) -> str:
    if material == "IN718":
        return f"IN718_QSR_x{scale}"
    return f"Ti_Al_1pct_QSR_x{scale}"


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def build_scale_dataset(material: str, spec: dict, scale: int) -> Path:
    src_root = Path(spec["source_dataset"])
    if not (src_root / "dataset_info.json").exists():
        raise FileNotFoundError(f"Missing source dataset_info.json: {src_root}")

    ds_name = dataset_name(material, scale)
    out_root = DATA_ROOT / ds_name
    counts: dict[str, dict[str, int]] = {}
    splits: dict[str, dict[str, str]] = {}

    for split in ("Train", "Val", "Test"):
        split_lower = split.lower()
        hr_out = out_root / split / "HR_Data"
        lr_out = out_root / split / "LR_Data"
        hr_out.mkdir(parents=True, exist_ok=True)
        lr_out.mkdir(parents=True, exist_ok=True)

        source_hr_files = sorted((src_root / split / "HR_Data").glob("*.npy"))
        if not source_hr_files:
            raise FileNotFoundError(f"No source HR npy files found in {src_root / split / 'HR_Data'}")

        n_hr = n_lr = 0
        for src_hr in source_hr_files:
            match = NAME_RE.match(src_hr.name)
            if match is None:
                raise ValueError(f"Unexpected source HR filename: {src_hr.name}")
            axis = match.group("axis").lower()
            block_id = match.group("id")
            hr_name = f"{ds_name}_{split_lower}_hr_{axis}_block_{block_id}.npy"
            lr_name = f"{ds_name}_{split_lower}_lr_{axis}_block_{block_id}.npy"
            dst_hr = hr_out / hr_name
            dst_lr = lr_out / lr_name
            _safe_symlink(src_hr.resolve(), dst_hr)
            n_hr += 1
            if not dst_lr.exists():
                hr = np.load(src_hr, mmap_mode="r")
                lr = np.asarray(hr[::scale, ::scale, :], dtype=np.float32)
                np.save(dst_lr, lr)
            n_lr += 1

        counts[split] = {"hr": n_hr, "lr": n_lr, "org": 0}
        splits[split] = {
            "HR_glob": str(hr_out / "*.npy"),
            "LR_glob": str(lr_out / "*.npy"),
        }

    source_info = json.loads((src_root / "dataset_info.json").read_text(encoding="utf-8"))
    patch_shape = source_info.get("patch_shape")
    info = {
        "dataset": ds_name,
        "patch_shape": patch_shape,
        "scale": int(scale),
        "symmetry": spec["symmetry"],
        "source_dataset": str(src_root),
        "construction": f"HR symlinks from source x4 dataset; LR generated as HR[::{scale}, ::{scale}].",
        "counts": counts,
        "splits": splits,
    }
    write_json(out_root / "dataset_info.json", info)
    return out_root


def make_scale_config(material: str, spec: dict, scale: int, dataset_root: Path) -> dict:
    current_cfg = json.loads((Path(spec["source_run"]) / "config_new.json").read_text(encoding="utf-8"))
    legacy_cfg = json.loads(
        (Path(spec["legacy_scale_runs"][scale]) / "config_new.json").read_text(encoding="utf-8")
    )
    cfg = json.loads(json.dumps(current_cfg))

    cfg["dataset_root"] = str(dataset_root)
    cfg["seed"] = 42
    cfg["scale"] = [scale, scale]
    cfg["upsample_factor"] = [scale, scale]
    cfg["symmetry_group"] = spec["symmetry"]
    cfg["crystal"] = spec["crystal"]
    cfg["embedding_mode"] = "direct_reynolds"
    cfg["max_harmonic_l"] = int(spec["harmonic_l"])
    cfg["embedding_metric_calibration"] = "isometric"
    cfg["cluster_source"] = "feature"
    cfg["cluster_feature_l2_threshold"] = current_cfg.get(
        "cluster_feature_l2_threshold",
        float(np.deg2rad(float(current_cfg.get("cluster_threshold_deg", 2.0)))),
    )
    cfg["cluster_threshold_deg"] = current_cfg.get("cluster_threshold_deg", 2.0)

    cfg["model"] = dict(current_cfg.get("model", {}))
    cfg["model"]["type"] = "iso_embedding_ocrp"
    cfg["model"]["model_module"] = "models.SR_4x4_from_4x1_ocrp_anchorless"
    cfg["model"]["model_class"] = legacy_cfg["model"]["model_class"]

    for key in (
        "batch_size",
        "min_free_cuda_gb",
        "ocrp_router_chunk_size",
        "ocrp_proposal_chunk_size",
        "ocrp_proposal_token_chunk_size",
    ):
        if key in legacy_cfg:
            cfg[key] = legacy_cfg[key]

    cfg.pop("checkpoints_dir", None)
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
    return cfg


def exp_dir(material: str, spec: dict, scale: int) -> Path:
    return RUN_ROOT / material / f"{spec['prefix']}_x{scale}_s42"


def write_run_script(exp_path: Path, scale: int) -> None:
    rel = exp_path.relative_to(ROOT)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        'GPU_ID="${1:-0}"',
        f'EXP_DIR="{rel}"',
        'mkdir -p "$EXP_DIR/logs"',
        'export CUDA_VISIBLE_DEVICES="$GPU_ID"',
        f"export MPLCONFIGDIR=/tmp/direct-reynolds-scaling-x{scale}-s42-gpu${{GPU_ID}}",
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
    run_path = exp_path / "run.sh"
    run_path.write_text("\n".join(lines), encoding="utf-8")
    make_executable(run_path)


def prepare_experiments() -> list[tuple[str, int, Path, int]]:
    jobs: list[tuple[str, int, Path, int]] = []
    gpu_id = 0
    for material, spec in MATERIALS.items():
        for scale in SCALES:
            dataset_root = build_scale_dataset(material, spec, scale)
            out_exp = exp_dir(material, spec, scale)
            out_exp.mkdir(parents=True, exist_ok=True)
            (out_exp / "logs").mkdir(exist_ok=True)
            (out_exp / "checkpoints").mkdir(exist_ok=True)
            write_json(out_exp / "config_new.json", make_scale_config(material, spec, scale, dataset_root))
            write_run_script(out_exp, scale)
            if not (out_exp / "STATUS").exists():
                (out_exp / "STATUS").write_text("PENDING\n", encoding="utf-8")
            if not (out_exp / "ISTATUS").exists():
                (out_exp / "ISTATUS").write_text("IPENDING\n", encoding="utf-8")
            jobs.append((material, scale, out_exp, gpu_id))
            gpu_id += 1
    return jobs


def write_master(jobs: list[tuple[str, int, Path, int]]) -> Path:
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    master = LAUNCH_ROOT / "run_scaling_seed42_train_infer.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        'RUN_ID="${RUN_ID:-scaling_direct_reynolds_$(date +%Y%m%d_%H%M%S)}"',
        'echo "[master] RUN_ID=$RUN_ID"',
        "pids=()",
        "",
    ]
    for material, scale, path, gpu in jobs:
        rel = path.relative_to(ROOT)
        lines.extend(
            [
                "(",
                "  set -euo pipefail",
                f'  echo "[job] {material} x{scale} train on gpu{gpu}"',
                f'  if [ "$(cat {rel}/STATUS 2>/dev/null)" != "DONE" ]; then',
                f'    echo RUNNING > {rel}/STATUS',
                f'    bash {rel}/run.sh {gpu} > {rel}/logs/train_${{RUN_ID}}.log 2>&1',
                f'    echo TRAIN_DONE > {rel}/STATUS',
                "  fi",
                f'  echo "[job] {material} x{scale} inference on gpu{gpu}"',
                f'  if [ "$(cat {rel}/ISTATUS 2>/dev/null)" != "IDONE" ]; then',
                f'    echo IRUNNING > {rel}/ISTATUS',
                f"    CUDA_VISIBLE_DEVICES={gpu} {PYTHON} inference/infer_iso_embedding_sr_attn.py \\",
                f"      --exp_dir {rel} \\",
                "      --config config_new.json \\",
                "      --checkpoint best_model.pt \\",
                "      --split Test \\",
                f"      --out_dir {rel}/inference/test_best \\",
                "      --skip_ipf \\",
                f"      --gpu_ids {gpu} > {rel}/logs/infer_test_best_${{RUN_ID}}.log 2>&1",
                f'    echo IDONE > {rel}/ISTATUS',
                f'    echo DONE > {rel}/STATUS',
                "  fi",
                ") &",
                'pids+=("$!")',
                "",
            ]
        )
    lines.extend(
        [
            'for pid in "${pids[@]}"; do wait "$pid"; done',
            'echo "[master] scaling train/infer jobs complete"',
            "",
        ]
    )
    master.write_text("\n".join(lines), encoding="utf-8")
    make_executable(master)
    return master


def main() -> None:
    jobs = prepare_experiments()
    master = write_master(jobs)
    print("Prepared corrected direct-Reynolds-isometric scale jobs:")
    for material, scale, path, gpu in jobs:
        print(f"  {material} x{scale}: {path.relative_to(ROOT)} on gpu{gpu}")
    print(f"Master launcher: {master.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
