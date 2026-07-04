#!/usr/bin/env python3
"""Create lean five-seed direct-Reynolds-isometric OCRP sweep jobs."""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/home/umang/miniconda3/envs/material/bin/python"
SEEDS = [42, 43, 44, 45, 46]
GPUS = [0, 1, 2, 3, 4, 5, 6]

MATERIALS = {
    "IN718": {
        "base": ROOT / "experiments/IN718/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l4_s42",
        "root": ROOT / "experiments/IN718/direct_reynolds_isometric_seed_runs",
        "prefix": "ocrp_direct_reynolds_isometric_l4",
        "estimate": 3.2,
        "mpl": "matplotlib-in718-direct-reynolds-isometric",
    },
    "Ti_Al_1pct": {
        "base": ROOT / "experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42",
        "root": ROOT / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs",
        "prefix": "ocrp_direct_reynolds_isometric_l6",
        "estimate": 3.5,
        "mpl": "matplotlib-ti-direct-reynolds-isometric",
    },
}


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seed_dir(spec: dict, seed: int) -> Path:
    return spec["root"] / f"{spec['prefix']}_s{seed}"


def make_config(base_cfg: dict, seed: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.pop("checkpoints_dir", None)
    cfg["seed"] = int(seed)
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


def write_run_script(exp_dir: Path, material: str, seed: int, mpl_name: str) -> None:
    rel = exp_dir.relative_to(ROOT)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        'GPU_ID="${1:-0}"',
        f'EXP_DIR="{rel}"',
        'mkdir -p "$EXP_DIR/logs"',
        'export CUDA_VISIBLE_DEVICES="$GPU_ID"',
        f"export MPLCONFIGDIR=/tmp/{mpl_name}-s{seed}",
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
    path = exp_dir / "run.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    make_executable(path)


def setup_seed_dirs() -> list[tuple[str, int, Path, float]]:
    jobs: list[tuple[str, int, Path, float]] = []
    for material, spec in MATERIALS.items():
        base = spec["base"]
        base_cfg = json.loads((base / "config_new.json").read_text(encoding="utf-8"))
        for seed in SEEDS:
            exp_dir = seed_dir(spec, seed)
            exp_dir.mkdir(parents=True, exist_ok=True)
            write_json(exp_dir / "config_new.json", make_config(base_cfg, seed))
            write_run_script(exp_dir, material, seed, spec["mpl"])
            (exp_dir / "logs").mkdir(exist_ok=True)
            (exp_dir / "checkpoints").mkdir(exist_ok=True)
            if seed == 42:
                src_ckpt = base / "checkpoints/best_model.pt"
                dst_ckpt = exp_dir / "checkpoints/best_model.pt"
                if src_ckpt.exists() and not dst_ckpt.exists():
                    shutil.copy2(src_ckpt, dst_ckpt)
                src_hist = base / "checkpoints/history.json"
                dst_hist = exp_dir / "checkpoints/history.json"
                if src_hist.exists() and not dst_hist.exists():
                    shutil.copy2(src_hist, dst_hist)
                (exp_dir / "STATUS").write_text("DONE\n", encoding="utf-8")
            else:
                status = exp_dir / "STATUS"
                if not status.exists():
                    status.write_text("PENDING\n", encoding="utf-8")
                jobs.append((material, seed, exp_dir, float(spec["estimate"])))
    return jobs


def distribute_jobs(jobs: list[tuple[str, int, Path, float]]) -> dict[int, list[tuple[str, int, Path, float]]]:
    queues = {gpu: [] for gpu in GPUS}
    loads = {gpu: 0.0 for gpu in GPUS}
    for job in sorted(jobs, key=lambda item: item[3], reverse=True):
        gpu = min(GPUS, key=lambda g: loads[g])
        queues[gpu].append(job)
        loads[gpu] += job[3]
    return queues


def write_train_launch(launch_root: Path, queues: dict[int, list[tuple[str, int, Path, float]]]) -> list[Path]:
    scripts = []
    for gpu, queue in queues.items():
        path = launch_root / f"train_gpu{gpu}.sh"
        lines = ["#!/usr/bin/env bash", "set -u", f"cd {ROOT}", ""]
        for material, seed, exp_dir, _ in queue:
            rel = exp_dir.relative_to(ROOT)
            lines.extend(
                [
                    f'if [ "$(cat {rel}/STATUS 2>/dev/null)" = DONE ]; then',
                    f'  echo "SKIP train {material} s{seed}"',
                    "else",
                    f'  echo "[$(date +%F_%T)] TRAIN {material} s{seed} gpu{gpu}"',
                    f'  bash {rel}/run.sh {gpu} > {rel}/logs/train.log 2>&1 '
                    f'&& echo DONE > {rel}/STATUS || echo FAILED > {rel}/STATUS',
                    "fi",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        make_executable(path)
        scripts.append(path)
    return scripts


def write_infer_launch(launch_root: Path) -> list[Path]:
    jobs: list[tuple[str, int, Path, float]] = []
    for material, spec in MATERIALS.items():
        for seed in SEEDS:
            jobs.append((material, seed, seed_dir(spec, seed), 1.0))
    queues = distribute_jobs(jobs)
    scripts = []
    for gpu, queue in queues.items():
        path = launch_root / f"infer_gpu{gpu}.sh"
        lines = ["#!/usr/bin/env bash", "set -u", f"cd {ROOT}", ""]
        for material, seed, exp_dir, _ in queue:
            rel = exp_dir.relative_to(ROOT)
            out_dir = rel / "inference/test_best"
            lines.extend(
                [
                    f'if [ "$(cat {rel}/ISTATUS 2>/dev/null)" = IDONE ]; then',
                    f'  echo "SKIP infer {material} s{seed}"',
                    "else",
                    f'  echo "[$(date +%F_%T)] INFER {material} s{seed} gpu{gpu}"',
                    f"  {PYTHON} -m inference.infer_iso_embedding_sr_attn "
                    f"--exp_dir {rel} --config config_new.json --checkpoint best_model.pt "
                    f"--split Test --out_dir {out_dir} --gpu_ids {gpu} --skip_ipf "
                    f"> {rel}/logs/infer_test_best.log 2>&1 "
                    f"&& echo IDONE > {rel}/ISTATUS || echo IFAILED > {rel}/ISTATUS",
                    "fi",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        make_executable(path)
        scripts.append(path)
    return scripts


def status_paths(kind: str) -> list[Path]:
    paths = []
    for spec in MATERIALS.values():
        for seed in SEEDS:
            paths.append(seed_dir(spec, seed) / kind)
    return paths


def write_master(launch_root: Path, train_scripts: list[Path], infer_scripts: list[Path]) -> Path:
    master = launch_root / "run_train_infer_aggregate.sh"
    train_statuses = " ".join(str(p.relative_to(ROOT)) for p in status_paths("STATUS"))
    infer_statuses = " ".join(str(p.relative_to(ROOT)) for p in status_paths("ISTATUS"))
    train_jobs = " ".join(str(p.relative_to(ROOT)) for p in train_scripts)
    infer_jobs = " ".join(str(p.relative_to(ROOT)) for p in infer_scripts)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        f"TRAIN_JOBS=({train_jobs})",
        f"INFER_JOBS=({infer_jobs})",
        f"TRAIN_STATUS=({train_statuses})",
        f"INFER_STATUS=({infer_statuses})",
        "",
        "echo '[master] training missing seeds'",
        "pids=()",
        'for job in "${TRAIN_JOBS[@]}"; do bash "$job" > "$job.log" 2>&1 & pids+=("$!"); done',
        'for pid in "${pids[@]}"; do wait "$pid"; done',
        'for st in "${TRAIN_STATUS[@]}"; do',
        '  if [ "$(cat "$st" 2>/dev/null)" != DONE ]; then echo "[master] failed train status: $st"; exit 1; fi',
        "done",
        "",
        "echo '[master] running test inference from best checkpoints'",
        "pids=()",
        'for job in "${INFER_JOBS[@]}"; do bash "$job" > "$job.log" 2>&1 & pids+=("$!"); done',
        'for pid in "${pids[@]}"; do wait "$pid"; done',
        'for st in "${INFER_STATUS[@]}"; do',
        '  if [ "$(cat "$st" 2>/dev/null)" != IDONE ]; then echo "[master] failed infer status: $st"; exit 1; fi',
        "done",
        "",
        "echo '[master] aggregating metrics'",
        f"{PYTHON} analysis/evaluate_direct_reynolds_isometric_seed_sweep.py",
        "echo '[master] done'",
        "",
    ]
    master.write_text("\n".join(lines), encoding="utf-8")
    make_executable(master)
    return master


def main() -> None:
    jobs = setup_seed_dirs()
    launch_root = ROOT / "experiments/direct_reynolds_isometric_seed_sweep/_launch"
    launch_root.mkdir(parents=True, exist_ok=True)
    train_scripts = write_train_launch(launch_root, distribute_jobs(jobs))
    infer_scripts = write_infer_launch(launch_root)
    master = write_master(launch_root, train_scripts, infer_scripts)
    print(f"Prepared {len(jobs)} new training jobs plus copied seed 42 checkpoints.")
    print(f"Master script: {master.relative_to(ROOT)}")
    for gpu, queue in distribute_jobs(jobs).items():
        desc = ", ".join(f"{mat}s{seed}" for mat, seed, _, _ in queue) or "-"
        print(f"  gpu{gpu}: {desc}")


if __name__ == "__main__":
    main()
