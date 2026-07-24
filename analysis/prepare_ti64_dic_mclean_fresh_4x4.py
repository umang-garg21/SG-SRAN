#!/usr/bin/env python3
"""Prepare and optionally launch fresh 4x4 Ti64 DIC McLean experiments.

This experiment uses the dataset's existing Train/Val/Test split when present.
It creates fresh configs for all learnable baselines plus the selected OCRP
direct-Reynolds isometric feature-distance model, then writes tmux worker
scripts and an automatic final evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")
DATASET_ROOT = Path("/data/home/umang/Materials/Materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4")
EXP_ROOT = ROOT / "experiments" / "Ti64_DIC_Mclean_fresh_4x4_existing_split"
OUT_DIR = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4"
BATCH_SIZE = 5


METHODS: list[dict[str, Any]] = [
    {
        "name": "EDSR",
        "slug": "edsr_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/edsr_4x4_01/config.json",
        "config": "config.json",
        "gpu": 0,
    },
    {
        "name": "QEDSR",
        "slug": "qedsr_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/qedsr_4x4_01/config.json",
        "config": "config.json",
        "gpu": 0,
    },
    {
        "name": "Atindama",
        "slug": "atindama_inpainting_4x4_fresh_ti64_dic_mclean",
        "kind": "atindama",
        "source": "Ti_Al_1pct/atindama_inpainting_4x4_01/config.json",
        "config": "config.json",
        "gpu": 1,
    },
    {
        "name": "Q-RBSA-adapted",
        "slug": "qrbsa_adapted_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01/config.json",
        "config": "config.json",
        "gpu": 2,
    },
    {
        "name": "RCAN",
        "slug": "rcan_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/rcan_4x4_300ep_01/config.json",
        "config": "config.json",
        "gpu": 3,
    },
    {
        "name": "SAN",
        "slug": "san_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/san_4x4_300ep_01/config.json",
        "config": "config.json",
        "gpu": 4,
    },
    {
        "name": "HAN",
        "slug": "han_4x4_fresh_ti64_dic_mclean",
        "kind": "jangid",
        "source": "Ti_Al_1pct/han_4x4_300ep_01/config.json",
        "config": "config.json",
        "gpu": 5,
    },
    {
        "name": "OCRP-direct-Reynolds-isometric-L6",
        "slug": "ocrp_direct_reynolds_isometric_l6_4x4_fresh_ti64_dic_mclean",
        "kind": "ocrp",
        "source": "Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42/config_new.json",
        "config": "config_new.json",
        "gpu": 6,
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def existing_split_counts(dataset_root: Path) -> dict[str, int]:
    info = read_json(dataset_root / "dataset_info.json")
    counts: dict[str, int] = {}
    for split in ("Train", "Val", "Test"):
        split_dir = dataset_root / split / "HR_Data"
        files = sorted(split_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No HR NPY files found for existing split {split}: {split_dir}")
        counts[split] = len(files)
    info_counts = {
        split: int(info.get("counts", {}).get(split, {}).get("hr", counts[split]))
        for split in ("Train", "Val", "Test")
    }
    if info_counts != counts:
        raise RuntimeError(f"dataset_info split counts {info_counts} differ from files on disk {counts}")
    return counts


def strip_split_overrides(cfg: dict[str, Any]) -> None:
    for key in (
        "fewshot",
        "fewshot_mode",
        "fewshot_take_first",
        "take_first",
        "train_take_first",
        "val_take_first",
        "test_take_first",
        "zeroshot",
        "zeroshot_mode",
        "source_dataset_root",
        "target_dataset_root",
    ):
        cfg.pop(key, None)


def patch_common_config(cfg: dict[str, Any], *, method: dict[str, Any], split_counts: dict[str, int]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    strip_split_overrides(cfg)
    cfg["dataset_root"] = str(DATASET_ROOT)
    cfg["symmetry_group"] = "D6h"
    cfg["symmetry"] = "D6h"
    cfg["crystal"] = "hcp"
    cfg["scale"] = [4, 4]
    cfg["upsample_factor"] = [4, 4]
    cfg["batch_size"] = BATCH_SIZE
    cfg["eval_batch_size"] = BATCH_SIZE
    cfg["drop_last"] = True
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["num_workers"] = int(cfg.get("num_workers", 8))
    cfg["preload"] = bool(cfg.get("preload", True))
    cfg["preload_torch"] = bool(cfg.get("preload_torch", True))
    cfg["fresh_ti64_dic_mclean_4x4"] = {
        "dataset_root": str(DATASET_ROOT),
        "split_policy": "Use existing Train/Val/Test split present in the dataset.",
        "split_counts": split_counts,
        "source_config": str(ROOT / "experiments" / method["source"]),
        "material_symmetry": "D6h",
    }
    return cfg


def patch_baseline_config(cfg: dict[str, Any], *, method: dict[str, Any], split_counts: dict[str, int]) -> dict[str, Any]:
    cfg = patch_common_config(cfg, method=method, split_counts=split_counts)
    cfg["save_every"] = 0
    cfg["viz_every"] = 0
    cfg["save_last_checkpoint"] = False
    cfg["save_epoch_checkpoints"] = False
    cfg["final_viz"] = False
    cfg["plot_loss_curves"] = True
    cfg.setdefault("logging", {})
    cfg["logging"]["save_best_only"] = True
    cfg["logging"]["tensorboard"] = False
    cfg["logging"]["val_freq"] = 1
    return cfg


def patch_ocrp_config(cfg: dict[str, Any], *, method: dict[str, Any], exp_dir: Path, split_counts: dict[str, int]) -> dict[str, Any]:
    cfg = patch_common_config(cfg, method=method, split_counts=split_counts)
    cfg["checkpoints_dir"] = str(exp_dir / "checkpoints")
    cfg["save_every"] = 1
    cfg["viz_every"] = 1
    cfg["save_last_checkpoint"] = True
    cfg["save_epoch_checkpoints"] = True
    cfg["final_viz"] = True
    cfg["plot_loss_curves"] = True
    cfg["viz_ref_dir"] = "ALL"
    cfg["viz_sample_index"] = 0
    cfg["viz_sample_key"] = None
    cfg.setdefault("logging", {})
    cfg["logging"]["save_best_only"] = False
    cfg["logging"]["tensorboard"] = False
    cfg["logging"]["val_freq"] = 1
    return cfg


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def train_command(run: dict[str, Any]) -> str:
    exp_dir = Path(run["exp_dir"])
    cfg = run["config_name"]
    gpu = str(run["gpu"])
    if run["kind"] == "atindama":
        parts = [
            PYTHON,
            ROOT / "training" / "train_atindama_inpainting.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--gpu",
            gpu,
        ]
    elif run["kind"] == "ocrp":
        parts = [
            PYTHON,
            ROOT / "training" / "train_iso_embedding_ocrp.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--gpu_ids",
            gpu,
        ]
    else:
        parts = [
            PYTHON,
            ROOT / "training" / "train_jangid_baseline.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--skip_viz",
            "--gpu",
            gpu,
        ]
    return shell_join(parts)


def infer_command(run: dict[str, Any]) -> str:
    exp_dir = Path(run["exp_dir"])
    cfg = run["config_name"]
    out_dir = exp_dir / "inference" / "test_best"
    gpu = str(run["gpu"])
    if run["kind"] == "atindama":
        parts = [
            PYTHON,
            ROOT / "inference" / "infer_atindama_inpainting.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--checkpoint",
            "best_model.pt",
            "--split",
            "Test",
            "--out_dir",
            out_dir,
            "--max_visualizations",
            "0",
            "--gpu",
            gpu,
        ]
    elif run["kind"] == "ocrp":
        parts = [
            PYTHON,
            ROOT / "inference" / "infer_iso_embedding_sr_attn.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--checkpoint",
            "best_model.pt",
            "--split",
            "Test",
            "--out_dir",
            out_dir,
            "--skip_ipf",
            "--gpu_ids",
            gpu,
        ]
    else:
        parts = [
            PYTHON,
            ROOT / "inference" / "infer_jangid_baseline.py",
            "--exp_dir",
            exp_dir,
            "--config",
            cfg,
            "--checkpoint",
            "best_model.pt",
            "--split",
            "Test",
            "--out_dir",
            out_dir,
            "--max_visualizations",
            "0",
            "--gpu",
            gpu,
        ]
    return shell_join(parts)


def env_prefix(gpu: int) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        "MPLCONFIGDIR=/tmp/matplotlib "
        "NUMBA_CACHE_DIR=/tmp/numba"
    )


def write_worker_script(path: Path, runs: list[dict[str, Any]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"mkdir -p {shlex.quote(str(OUT_DIR / 'logs'))}",
        f"echo \"[$(date)] worker starting: {path.name}\"",
        "",
    ]
    for run in runs:
        exp_dir = Path(run["exp_dir"])
        summary = exp_dir / "inference" / "test_best" / "summary.json"
        best_model = exp_dir / "checkpoints" / "best_model.pt"
        logs = exp_dir / "logs"
        train_log = logs / "tmux_train.log"
        infer_log = logs / "tmux_infer_test_best.log"
        gpu = int(run["gpu"])
        lines.extend(
            [
                f"echo \"[$(date)] === {run['name']} ===\"",
                f"mkdir -p {shlex.quote(str(logs))}",
                f"if [[ -f {shlex.quote(str(summary))} ]]; then",
                f"  echo \"[$(date)] {run['name']}: summary already exists; skipping train/infer.\"",
                "else",
                f"  if [[ ! -f {shlex.quote(str(best_model))} ]]; then",
                f"    echo \"[$(date)] {run['name']}: training on GPU {gpu}\"",
                f"    {env_prefix(gpu)} {train_command(run)} > {shlex.quote(str(train_log))} 2>&1",
                f"    touch {shlex.quote(str(logs / 'train.done'))}",
                "  else",
                f"    echo \"[$(date)] {run['name']}: best checkpoint exists; skipping training.\"",
                "  fi",
                f"  echo \"[$(date)] {run['name']}: test inference on GPU {gpu}\"",
                f"  {env_prefix(gpu)} {infer_command(run)} > {shlex.quote(str(infer_log))} 2>&1",
                f"  touch {shlex.quote(str(logs / 'infer_test_best.done'))}",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            f"touch {shlex.quote(str(path.with_suffix('.done')))}",
            f"echo \"[$(date)] worker done: {path.name}\"",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    make_executable(path)


def write_finalizer(path: Path, manifest_path: Path, runs: list[dict[str, Any]]) -> None:
    summary_paths = [Path(run["exp_dir"]) / "inference" / "test_best" / "summary.json" for run in runs]
    checks = "\n".join(f"  [[ -f {shlex.quote(str(p))} ]] || missing=$((missing + 1))" for p in summary_paths)
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
mkdir -p {shlex.quote(str(OUT_DIR / 'logs'))}
echo "[$(date)] waiting for all test summaries"
while true; do
  missing=0
{checks}
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  echo "[$(date)] waiting: $missing summaries still missing"
  sleep 300
done
echo "[$(date)] all summaries present; running metric panel"
{env_prefix(6)} {shell_join([PYTHON, ROOT / 'analysis' / 'evaluate_ti64_dic_mclean_fresh_4x4.py', '--manifest', manifest_path, '--include-classical'])} > {shlex.quote(str(OUT_DIR / 'logs' / 'final_metrics.log'))} 2>&1
touch {shlex.quote(str(OUT_DIR / 'logs' / 'final_metrics.done'))}
echo "[$(date)] metric panel complete"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    make_executable(path)


def launch_tmux(manifest: dict[str, Any], *, session_prefix: str) -> None:
    scripts = manifest["tmux_scripts"]
    for item in scripts["workers"]:
        session = f"{session_prefix}_g{item['gpu']}"
        cmd = ["tmux", "new-session", "-d", "-s", session, str(item["script"])]
        subprocess.run(cmd, check=True)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", f"{session_prefix}_finalize", str(scripts["finalizer"])],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Launch tmux workers after writing configs/scripts.")
    parser.add_argument("--session-prefix", default="qsr_ti64fresh", help="tmux session prefix.")
    args = parser.parse_args()

    split_counts = existing_split_counts(DATASET_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for method in METHODS:
        src_cfg_path = ROOT / "experiments" / method["source"]
        if not src_cfg_path.exists():
            raise FileNotFoundError(src_cfg_path)
        exp_dir = EXP_ROOT / method["slug"]
        cfg_path = exp_dir / method["config"]
        src_cfg = read_json(src_cfg_path)
        if method["kind"] == "ocrp":
            cfg = patch_ocrp_config(src_cfg, method=method, exp_dir=exp_dir, split_counts=split_counts)
        else:
            cfg = patch_baseline_config(src_cfg, method=method, split_counts=split_counts)
        write_json(cfg_path, cfg)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        runs.append(
            {
                "name": method["name"],
                "slug": method["slug"],
                "kind": method["kind"],
                "gpu": method["gpu"],
                "exp_dir": str(exp_dir),
                "config_name": method["config"],
                "config_path": str(cfg_path),
                "source_config": str(src_cfg_path),
                "checkpoint": str(exp_dir / "checkpoints" / "best_model.pt"),
                "inference_dir": str(exp_dir / "inference" / "test_best"),
                "summary": str(exp_dir / "inference" / "test_best" / "summary.json"),
            }
        )

    workers_dir = OUT_DIR / "tmux_workers"
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_gpu[int(run["gpu"])].append(run)

    worker_entries: list[dict[str, Any]] = []
    for gpu, gpu_runs in sorted(by_gpu.items()):
        script = workers_dir / f"worker_gpu{gpu}.sh"
        write_worker_script(script, gpu_runs)
        worker_entries.append({"gpu": gpu, "script": str(script), "runs": [r["name"] for r in gpu_runs]})

    manifest_path = OUT_DIR / "manifest.json"
    finalizer = workers_dir / "finalize_when_done.sh"
    manifest: dict[str, Any] = {
        "experiment": "Ti64_DIC_Mclean_fresh_4x4_existing_split",
        "created_by": Path(__file__).name,
        "root": str(ROOT),
        "dataset_root": str(DATASET_ROOT),
        "dataset_info": str(DATASET_ROOT / "dataset_info.json"),
        "split_policy": "Use the existing Train/Val/Test split already present in the dataset.",
        "split_counts": split_counts,
        "material_symmetry": "D6h",
        "scale": [4, 4],
        "experiment_root": str(EXP_ROOT),
        "analysis_out": str(OUT_DIR),
        "runs": runs,
        "tmux_scripts": {
            "workers": worker_entries,
            "finalizer": str(finalizer),
            "session_prefix": args.session_prefix,
        },
        "outputs": {
            "metrics_csv": str(OUT_DIR / "metrics.csv"),
            "metrics_json": str(OUT_DIR / "metrics.json"),
            "metrics_md": str(OUT_DIR / "metrics.md"),
            "best_by_metric_csv": str(OUT_DIR / "best_by_metric.csv"),
        },
    }
    write_json(manifest_path, manifest)
    write_finalizer(finalizer, manifest_path, runs)

    launcher = OUT_DIR / "launch_tmux.sh"
    launch_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for item in worker_entries:
        launch_lines.append(
            f"tmux new-session -d -s {shlex.quote(args.session_prefix + '_g' + str(item['gpu']))} {shlex.quote(item['script'])}"
        )
    launch_lines.append(
        f"tmux new-session -d -s {shlex.quote(args.session_prefix + '_finalize')} {shlex.quote(str(finalizer))}"
    )
    launcher.write_text("\n".join(launch_lines) + "\n")
    make_executable(launcher)

    print(f"Prepared experiment: {EXP_ROOT}")
    print(f"Using existing split counts: {split_counts}")
    print(f"Manifest: {manifest_path}")
    print(f"Launcher: {launcher}")
    for item in worker_entries:
        print(f"GPU {item['gpu']}: {', '.join(item['runs'])}")

    if args.launch:
        launch_tmux(manifest, session_prefix=args.session_prefix)
        print(f"Launched tmux sessions with prefix: {args.session_prefix}")


if __name__ == "__main__":
    main()
