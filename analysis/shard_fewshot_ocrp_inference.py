#!/usr/bin/env python3
"""Prepare and merge sharded few-shot OCRP inference runs.

This is intentionally narrow: it keeps the normal inference script untouched,
but gives it shard-specific dataset roots so expensive OCRP decoding can run
on multiple GPUs and be merged back into the standard ``inference/test_best``
layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"Refusing to replace existing path: {dst}")
    os.symlink(src, dst)


def _pair_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    parts = stem.split("_")
    axis = parts[-3]
    block = int(parts[-1])
    return axis, block


def _test_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    info = _read_json(dataset_root / "dataset_info.json")
    hr_files = sorted(Path(p) for p in Path().glob("__never_matches__"))
    # dataset_info globs may be absolute, but the few-shot roots are local and
    # deterministic. Prefer the actual shardable folders.
    hr_files = sorted((dataset_root / "Test" / "HR_Data").glob("*.npy"), key=_pair_key)
    lr_files = sorted((dataset_root / "Test" / "LR_Data").glob("*.npy"), key=_pair_key)
    if not hr_files or not lr_files:
        raise RuntimeError(f"No Test files found under {dataset_root}")

    lr_by_key = {_pair_key(p): p for p in lr_files}
    hr_by_key = {_pair_key(p): p for p in hr_files}
    keys = sorted(lr_by_key.keys() & hr_by_key.keys())
    if len(keys) != len(hr_files) or len(keys) != len(lr_files):
        raise RuntimeError(
            f"Unmatched LR/HR files in {dataset_root}: "
            f"{len(lr_files)} LR, {len(hr_files)} HR, {len(keys)} paired"
        )
    return [(lr_by_key[k], hr_by_key[k]) for k in keys]


def _split_ranges(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    step = int(math.ceil(n_items / n_shards))
    ranges = []
    for shard_idx in range(n_shards):
        start = shard_idx * step
        end = min(n_items, start + step)
        if start < end:
            ranges.append((start, end))
    return ranges


def prepare(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    pairs = _test_pairs(dataset_root)
    ranges = _split_ranges(len(pairs), int(args.num_shards))
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    if len(gpus) < len(ranges):
        raise ValueError(f"Need at least {len(ranges)} GPU ids, got {gpus}")

    shard_root = exp_dir / "inference" / "test_best_shards"
    scripts_dir = ROOT / "analysis" / "out" / "fewshot_ocrp_shards"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    base_info = _read_json(dataset_root / "dataset_info.json")
    base_cfg = _read_json(exp_dir / args.config)
    commands = []
    manifest = {
        "exp_dir": str(exp_dir),
        "dataset_root": str(dataset_root),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": "Test",
        "num_samples": len(pairs),
        "shards": [],
    }

    for shard_idx, (start, end) in enumerate(ranges):
        shard_name = f"shard_{shard_idx:02d}"
        shard_dataset = shard_root / shard_name / "dataset"
        shard_out = shard_root / shard_name / "out"
        for lr_src, hr_src in pairs[start:end]:
            _safe_symlink(lr_src.resolve(), shard_dataset / "Test" / "LR_Data" / lr_src.name)
            _safe_symlink(hr_src.resolve(), shard_dataset / "Test" / "HR_Data" / hr_src.name)

        info = dict(base_info)
        info["splits"] = dict(base_info["splits"])
        info["splits"]["Test"] = dict(base_info["splits"]["Test"])
        info["splits"]["Test"]["LR_glob"] = str(shard_dataset / "Test" / "LR_Data" / "*.npy")
        info["splits"]["Test"]["HR_glob"] = str(shard_dataset / "Test" / "HR_Data" / "*.npy")
        _write_json(shard_dataset / "dataset_info.json", info)

        cfg = dict(base_cfg)
        cfg["dataset_root"] = str(shard_dataset)
        cfg["batch_size"] = int(args.batch_size)
        cfg["num_workers"] = int(args.num_workers)
        cfg["inference_num_workers"] = int(args.num_workers)
        cfg_path = exp_dir / f"config_infer_{shard_name}.json"
        _write_json(cfg_path, cfg)

        gpu = gpus[shard_idx]
        command = [
            str(args.python),
            str(ROOT / "inference" / "infer_iso_embedding_sr_attn.py"),
            "--exp_dir",
            str(exp_dir),
            "--config",
            cfg_path.name,
            "--checkpoint",
            args.checkpoint,
            "--split",
            "Test",
            "--out_dir",
            str(shard_out),
            "--skip_ipf",
            "--gpu_ids",
            gpu,
        ]
        log_path = shard_root / shard_name / "infer.log"
        script_path = scripts_dir / f"run_ti64_ocrp_{shard_name}_gpu{gpu}.sh"
        script = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {ROOT}",
            " ".join(command) + f" > {log_path} 2>&1",
        ]
        script_path.write_text("\n".join(script) + "\n")
        script_path.chmod(0o755)
        commands.append((shard_idx, gpu, script_path))
        manifest["shards"].append(
            {
                "index": shard_idx,
                "start": start,
                "end": end,
                "gpu": gpu,
                "dataset_root": str(shard_dataset),
                "out_dir": str(shard_out),
                "log": str(log_path),
                "script": str(script_path),
            }
        )

    _write_json(shard_root / "manifest.json", manifest)
    print(f"Prepared {len(ranges)} shards for {len(pairs)} samples")
    print(f"Manifest: {shard_root / 'manifest.json'}")
    for shard_idx, gpu, script_path in commands:
        session = f"qsr_ti64_ocrp_shard_{shard_idx:02d}"
        print(f"tmux new-session -d -s {session} {script_path}")


def merge(args: argparse.Namespace) -> None:
    exp_dir = Path(args.exp_dir).resolve()
    shard_root = exp_dir / "inference" / "test_best_shards"
    manifest = _read_json(shard_root / "manifest.json")
    final_out = exp_dir / "inference" / "test_best"
    final_sr_dir = final_out / "sr_quaternions"
    final_sr_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for shard in manifest["shards"]:
        shard_summary = Path(shard["out_dir"]) / "summary.json"
        if not shard_summary.exists():
            raise FileNotFoundError(f"Missing shard summary: {shard_summary}")
        summary = _read_json(shard_summary)
        local_records = summary.get("records", [])
        expected = int(shard["end"]) - int(shard["start"])
        if len(local_records) != expected:
            raise RuntimeError(
                f"Shard {shard['index']} has {len(local_records)} records, expected {expected}"
            )
        for rec in local_records:
            global_sid = int(shard["start"]) + int(rec["sample_id"])
            merged = dict(rec)
            merged["sample_id"] = global_sid
            merged["shard_index"] = int(shard["index"])
            merged["shard_sample_id"] = int(rec["sample_id"])
            for kind in ("sr", "lr", "hr"):
                src_key = f"{kind}_npy"
                src = Path(rec[src_key])
                dst = final_sr_dir / f"sample_{global_sid:06d}_{kind}.npy"
                shutil.copy2(src, dst)
                merged[src_key] = str(dst)
            merged["ipf_png"] = None
            records.append(merged)

    records.sort(key=lambda r: int(r["sample_id"]))
    expected_ids = list(range(int(manifest["num_samples"])))
    observed_ids = [int(r["sample_id"]) for r in records]
    if observed_ids != expected_ids:
        raise RuntimeError("Merged sample ids are not contiguous")

    summary_path = final_out / "summary.json"
    _write_json(
        summary_path,
        {
            "exp_dir": str(exp_dir),
            "config": str(exp_dir / manifest["config"]),
            "checkpoint": str(
                exp_dir / "checkpoints" / manifest["checkpoint"]
                if not Path(manifest["checkpoint"]).is_absolute()
                else Path(manifest["checkpoint"])
            ),
            "split": "Test",
            "skip_ipf": True,
            "num_samples": len(records),
            "records": records,
            "merged_from_shards": manifest["shards"],
        },
    )
    print(f"Merged {len(records)} samples into {final_out}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "merge"])
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="config_new.json")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--python",
        default="/data/home/umang/miniconda3/envs/material/bin/python",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
