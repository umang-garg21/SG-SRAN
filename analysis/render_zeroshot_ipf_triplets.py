#!/usr/bin/env python3
"""Render LR/SR/HR IPF triptychs for saved zero-shot 4x4 outputs.

The zero-shot learned-baseline folders already store quaternion triplets under
``sr_quaternions``.  This script fills the matching ``ipf`` folders using the
same LR/SR/HR IPF-X/Y/Z layout used by the older OCRP zero-shot inference run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
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
class ZeroShotSet:
    key: str
    root: Path
    symmetry: str


ZERO_SHOT_SETS = [
    ZeroShotSet(
        key="CoNi",
        root=ROOT / "experiments/Zero_shot_performance_CoNi_x250",
        symmetry="Oh",
    ),
    ZeroShotSet(
        key="Ti7",
        root=ROOT / "experiments/Zero_shot_performance_Ti7_deformed",
        symmetry="D6h",
    ),
    ZeroShotSet(
        key="Ti64",
        root=ROOT / "experiments/Zero_shot_performance_Ti64_DIC_Mclean",
        symmetry="D6h",
    ),
]


def sample_ids(sr_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(sr_dir.glob("sample_*_sr.npy")):
        ids.append(int(path.name.split("_")[1]))
    return ids


def ipf_dir_for(sr_dir: Path) -> Path:
    return sr_dir.parent / "ipf"


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_triplet(sr_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"sample_{sample_id:06d}"
    lr = np.load(sr_dir / f"{stem}_lr.npy").astype(np.float32, copy=False)
    sr = np.load(sr_dir / f"{stem}_sr.npy").astype(np.float32, copy=False)
    hr = np.load(sr_dir / f"{stem}_hr.npy").astype(np.float32, copy=False)
    return lr, sr, hr


def render_folder(sr_dir: Path, symmetry_name: str, *, overwrite: bool, dpi: int) -> tuple[int, int]:
    sym = resolve_symmetry(symmetry_name)
    out_dir = ipf_dir_for(sr_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    ids = sample_ids(sr_dir)
    if not ids:
        print(f"  skip empty: {relative_label(sr_dir)}", flush=True)
        return rendered, skipped

    print(f"  {relative_label(sr_dir)}  n={len(ids)}", flush=True)
    for sample_id in ids:
        out_png = out_dir / f"sample_{sample_id:06d}_lr_sr_hr_ipf.png"
        if out_png.exists() and not overwrite:
            skipped += 1
            continue
        lr, sr, hr = load_triplet(sr_dir, sample_id)
        render_sr_hr_lr_side_by_side(
            sr_q_arr=sr,
            hr_q_arr=hr,
            lr_q_arr=lr,
            sym_class=sym,
            out_png=str(out_png),
            ref_dir="ALL",
            include_key=True,
            overwrite=True,
            format_input=True,
            dpi=dpi,
            pixels_per_image_pixel=1,
            include_row_labels=True,
        )
        rendered += 1
    return rendered, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets",
        nargs="*",
        default=[spec.key for spec in ZERO_SHOT_SETS],
        choices=[spec.key for spec in ZERO_SHOT_SETS],
        help="Zero-shot experiment sets to render.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing PNG files.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    selected = {spec.key: spec for spec in ZERO_SHOT_SETS}
    total_rendered = 0
    total_skipped = 0
    for key in args.sets:
        spec = selected[key]
        if not spec.root.exists():
            raise FileNotFoundError(spec.root)
        sr_dirs = sorted(spec.root.glob("**/sr_quaternions"))
        print(f"{spec.key}: {relative_label(spec.root)} ({spec.symmetry}), folders={len(sr_dirs)}", flush=True)
        for sr_dir in sr_dirs:
            rendered, skipped = render_folder(sr_dir, spec.symmetry, overwrite=args.overwrite, dpi=args.dpi)
            total_rendered += rendered
            total_skipped += skipped

    print(f"Done. rendered={total_rendered}, skipped_existing={total_skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
