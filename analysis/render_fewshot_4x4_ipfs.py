#!/usr/bin/env python3
"""Render LR/SR/HR IPF triptychs for few-shot 4x4 outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402


@dataclass(frozen=True)
class RenderTask:
    sr_dir: str
    ipf_dir: str
    sample_id: int
    symmetry: str
    dpi: int
    overwrite: bool


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _symmetry_for_dataset(dataset_root: Path) -> str:
    info = _read_json(dataset_root / "dataset_info.json")
    return str(info.get("symmetry", "Oh"))


def _sample_ids(sr_dir: Path) -> list[int]:
    ids = []
    for path in sorted(sr_dir.glob("sample_*_sr.npy")):
        ids.append(int(path.name.split("_")[1]))
    return ids


@lru_cache(maxsize=8)
def _resolved_symmetry(symmetry_name: str):
    return resolve_symmetry(symmetry_name)


def _load_triplet(sr_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"sample_{sample_id:06d}"
    lr = np.load(sr_dir / f"{stem}_lr.npy").astype(np.float32, copy=False)
    sr = np.load(sr_dir / f"{stem}_sr.npy").astype(np.float32, copy=False)
    hr = np.load(sr_dir / f"{stem}_hr.npy").astype(np.float32, copy=False)
    return lr, sr, hr


def _render_one(task: RenderTask) -> str:
    sr_dir = Path(task.sr_dir)
    ipf_dir = Path(task.ipf_dir)
    ipf_dir.mkdir(parents=True, exist_ok=True)
    out_png = ipf_dir / f"sample_{task.sample_id:06d}_lr_sr_hr_ipf.png"
    if out_png.exists() and not task.overwrite:
        return "skipped"

    lr, sr, hr = _load_triplet(sr_dir, task.sample_id)
    render_sr_hr_lr_side_by_side(
        sr_q_arr=sr,
        hr_q_arr=hr,
        lr_q_arr=lr,
        sym_class=_resolved_symmetry(task.symmetry),
        out_png=str(out_png),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=int(task.dpi),
        pixels_per_image_pixel=1,
        include_row_labels=True,
    )
    return "rendered"


def _selected_runs(manifest: dict, targets: set[str] | None, methods: set[str] | None) -> list[dict]:
    runs = []
    for run in manifest["runs"]:
        if targets and run["target_key"] not in targets and run["target_name"] not in targets:
            continue
        if methods and run["method"] not in methods:
            continue
        runs.append(run)
    return runs


def _build_tasks(
    runs: list[dict],
    *,
    overwrite: bool,
    dpi: int,
    max_samples: int | None,
) -> tuple[list[RenderTask], int, int, int]:
    tasks: list[RenderTask] = []
    completed_runs = 0
    pending_runs = 0
    existing_pngs = 0

    for run in runs:
        exp_dir = ROOT / run["experiment"]
        out_dir = exp_dir / "inference" / "test_best"
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            pending_runs += 1
            continue

        summary = _read_json(summary_path)
        expected = int(run.get("heldout_samples", summary.get("num_samples", 0)))
        observed = int(summary.get("num_samples", 0))
        if observed != expected:
            pending_runs += 1
            print(
                f"pending partial: {run['target_name']} :: {run['method']} "
                f"({observed}/{expected})",
                flush=True,
            )
            continue

        sr_dir = out_dir / "sr_quaternions"
        ids = _sample_ids(sr_dir)
        if max_samples is not None:
            ids = ids[: int(max_samples)]
        if not ids:
            pending_runs += 1
            continue

        completed_runs += 1
        symmetry = _symmetry_for_dataset(ROOT / run["fewshot_dataset_root"])
        ipf_dir = out_dir / "ipf"
        missing = 0
        for sample_id in ids:
            out_png = ipf_dir / f"sample_{sample_id:06d}_lr_sr_hr_ipf.png"
            if out_png.exists() and not overwrite:
                existing_pngs += 1
                continue
            missing += 1
            tasks.append(
                RenderTask(
                    sr_dir=str(sr_dir),
                    ipf_dir=str(ipf_dir),
                    sample_id=int(sample_id),
                    symmetry=symmetry,
                    dpi=int(dpi),
                    overwrite=bool(overwrite),
                )
            )
        print(
            f"{run['target_name']} :: {run['method']} "
            f"n={len(ids)} missing_ipf={missing} ({symmetry})",
            flush=True,
        )

    return tasks, completed_runs, pending_runs, existing_pngs


def render_once(args: argparse.Namespace) -> tuple[int, int, int, int, int]:
    manifest = _read_json(ROOT / args.manifest)
    targets = set(args.targets or []) or None
    methods = set(args.methods or []) or None
    runs = _selected_runs(manifest, targets, methods)
    tasks, completed_runs, pending_runs, existing_pngs = _build_tasks(
        runs,
        overwrite=bool(args.overwrite),
        dpi=int(args.dpi),
        max_samples=args.max_samples,
    )

    rendered = 0
    skipped = 0
    failed = 0
    if tasks:
        print(
            f"Rendering {len(tasks)} IPF PNGs with jobs={int(args.jobs)} "
            f"(completed_runs={completed_runs}, pending_runs={pending_runs})",
            flush=True,
        )
        if int(args.jobs) <= 1:
            for task in tasks:
                try:
                    status = _render_one(task)
                    rendered += int(status == "rendered")
                    skipped += int(status == "skipped")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"FAILED {task.ipf_dir} sample={task.sample_id}: {exc}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
                future_to_task = {pool.submit(_render_one, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        status = future.result()
                        rendered += int(status == "rendered")
                        skipped += int(status == "skipped")
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        print(f"FAILED {task.ipf_dir} sample={task.sample_id}: {exc}", flush=True)
    else:
        print(
            f"No missing IPFs right now "
            f"(completed_runs={completed_runs}, pending_runs={pending_runs}, existing={existing_pngs})",
            flush=True,
        )

    print(
        f"Pass complete: rendered={rendered}, skipped={skipped}, failed={failed}, "
        f"existing={existing_pngs}, completed_runs={completed_runs}, pending_runs={pending_runs}",
        flush=True,
    )
    return rendered, skipped, failed, completed_runs, pending_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="analysis/out/fewshot_4x4_manifest.json",
        help="Few-shot manifest relative to repo root.",
    )
    parser.add_argument("--targets", nargs="*", help="Target keys/names to render.")
    parser.add_argument("--methods", nargs="*", help="Method names to render.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--watch", action="store_true", help="Poll and render newly completed runs.")
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        _, _, failed, completed_runs, pending_runs = render_once(args)
        if failed:
            return 1
        if not args.watch:
            return 0
        if pending_runs == 0:
            # One final immediate pass verifies that all newly completed runs
            # have no missing PNGs, then the next loop would be a no-op.
            _, _, failed, _, pending_runs = render_once(args)
            return 1 if failed else 0
        print(
            f"Waiting {int(args.poll_seconds)}s for {pending_runs} pending run(s) to finish...",
            flush=True,
        )
        time.sleep(int(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
