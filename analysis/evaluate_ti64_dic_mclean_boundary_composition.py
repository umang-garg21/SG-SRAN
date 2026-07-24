#!/usr/bin/env python3
"""Run paper-style boundary-window composition PRF on Ti64 DIC McLean full train."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_boundary_spurious_composition as comp  # noqa: E402
import evaluate_ti64_dic_mclean_fresh_4x4 as eval_ti64  # noqa: E402


DEFAULT_PREFIX = "boundary_composition_prf_8nbr_5deg_full_train"


def parse_gpus(gpus_arg: str) -> list[int]:
    if gpus_arg == "auto":
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    return [int(item) for item in gpus_arg.split(",") if item.strip()]


def build_jobs(manifest: dict[str, Any], include_classical: bool) -> list[dict[str, Any]]:
    completed = [run for run in manifest["runs"] if eval_ti64.summary_path(run).exists()]
    ref_run = eval_ti64.choose_reference_run(completed)
    ref_dir = eval_ti64.sr_quat_dir(ref_run)
    jobs: list[dict[str, Any]] = []

    methods: OrderedDict[str, Path | None] = OrderedDict()
    if include_classical:
        methods.update(eval_ti64.CLASSICAL)
    for run in completed:
        methods[run["name"]] = eval_ti64.sr_quat_dir(run)

    for method, method_dir in methods.items():
        jobs.append(
            {
                "dataset": manifest["experiment"],
                "dataset_label": "Ti64 DIC McLean fresh 4x4",
                "symmetry": manifest.get("material_symmetry", "D6h"),
                "ref_dir": str(ref_dir),
                "method": method,
                "method_dir": None if method_dir is None else str(method_dir),
            }
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "experiments/Ti64_DIC_Mclean_fresh_4x4_full_Train/analysis/current_models_manifest.json",
    )
    parser.add_argument("--include-classical", action="store_true", default=True)
    parser.add_argument("--threshold-deg", type=float, default=5.0)
    parser.add_argument("--chunk-centers", type=int, default=8192)
    parser.add_argument("--gpus", default="auto", help="'auto', comma-separated CUDA ids, or 'cpu'")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--out-prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    manifest = eval_ti64.load_manifest(args.manifest)
    out_dir = Path(manifest["analysis_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(manifest, args.include_classical)
    if not jobs:
        raise RuntimeError("No jobs to run.")

    if args.gpus == "cpu":
        devices = ["cpu"]
    else:
        gpu_ids = parse_gpus(args.gpus)
        if not gpu_ids:
            raise RuntimeError("CUDA is not available; pass --gpus cpu for CPU execution.")
        devices = [f"cuda:{gid}" for gid in gpu_ids]

    workers = min(len(jobs), len(devices) if args.max_workers is None else args.max_workers)
    worker_devices = [devices[i % len(devices)] for i in range(len(jobs))]
    print(
        f"Running {len(jobs)} boundary-composition jobs with {workers} worker(s) on {devices}; "
        f"threshold={args.threshold_deg} deg",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    if workers == 1:
        for job, device in zip(jobs, worker_devices):
            result = comp.run_job(job, device, args.threshold_deg, args.chunk_centers, args.limit_samples)
            summaries.append(result["summary"])
            per_sample.extend(result["per_sample"])
            print(
                f"done {job['method']}: precision={result['summary']['composition_precision']:.6f}, "
                f"recall={result['summary']['composition_recall']:.6f}, "
                f"f1={result['summary']['composition_f1']:.6f}",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    comp.run_job,
                    job,
                    device,
                    args.threshold_deg,
                    args.chunk_centers,
                    args.limit_samples,
                ): job
                for job, device in zip(jobs, worker_devices)
            }
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                summaries.append(result["summary"])
                per_sample.extend(result["per_sample"])
                print(
                    f"done {job['method']}: precision={result['summary']['composition_precision']:.6f}, "
                    f"recall={result['summary']['composition_recall']:.6f}, "
                    f"f1={result['summary']['composition_f1']:.6f}",
                    flush=True,
                )

    summaries.sort(key=lambda r: (-r["composition_f1"], r["spurious_rate"], r["method"]))
    per_sample.sort(key=lambda r: (r["method"], r["sample_id"]))

    csv_path = out_dir / f"{args.out_prefix}.csv"
    sample_csv_path = out_dir / f"{args.out_prefix}_per_sample.csv"
    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"
    comp.write_csv(csv_path, summaries)
    comp.write_csv(sample_csv_path, per_sample)
    comp.write_markdown(md_path, summaries)
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_manifest": str(args.manifest),
                "protocol": {
                    "boundary_connectivity": 8,
                    "boundary_threshold_deg": args.threshold_deg,
                    "match_threshold_deg": args.threshold_deg,
                    "window_radius_px": 1,
                    "edge_wrapping": "none",
                    "counting": "overlapping GT boundary windows count separately",
                    "spurious_rate_higher_is_worse": True,
                    "precision": "1 - spurious_rate",
                    "recall": "GT orientation groups recovered / GT orientation groups",
                    "f1": "harmonic mean of composition precision and group-level recall",
                },
                "rows": summaries,
                "per_sample": per_sample,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {sample_csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
