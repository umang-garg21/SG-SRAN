#!/usr/bin/env python3
"""Render first-test-sample IPF-Z comparison for the fresh Ti64 DIC McLean run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v4" / "evals"
for path in (ROOT, EVAL_DIR, ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import metric_panel_hardened as mph  # noqa: E402
from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb  # noqa: E402


CLASSICAL = ("Nearest", "Bicubic", "SLERP", "Symm-SLERP")
LEARNED_ORDER = (
    "Atindama",
    "EDSR",
    "QEDSR",
    "Q-RBSA-adapted",
    "RCAN",
    "SAN",
    "HAN",
    "OCRP-direct-Reynolds-isometric-L6",
)
DISPLAY_NAME = {
    "Q-RBSA-adapted": "Q-RBSA",
    "OCRP-direct-Reynolds-isometric-L6": "OCRP",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _sr_dir(run: dict[str, Any]) -> Path:
    return Path(run["inference_dir"]) / "sr_quaternions"


def _load_q(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion array at {path}, got {arr.shape}")
    if arr.shape[-1] == 4:
        return arr
    if arr.shape[0] == 4:
        return np.moveaxis(arr, 0, -1).astype(np.float32, copy=False)
    raise ValueError(f"No quaternion axis found at {path}: {arr.shape}")


def _ipfz(q: np.ndarray, sym) -> np.ndarray:
    q_fmt = format_quaternions(
        q,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=sym,
        to_quat_first=False,
    )
    rgb = np.asarray(render_ipf_rgb(q_fmt, sym, ref_dir="Z"), dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _nearest_to_hw(rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if oh % h != 0 or ow % w != 0:
        raise ValueError(f"Cannot nearest-expand {rgb.shape} to {(oh, ow)}")
    return np.repeat(np.repeat(rgb, oh // h, axis=0), ow // w, axis=1)


def _mean_deg(sr: np.ndarray, hr: np.ndarray, ops: np.ndarray) -> float:
    return float(np.mean(mph.misorientation_fast(sr, hr, ops)))


def render(manifest_path: Path, sample_id: int, out_dir: Path, dpi: int) -> tuple[Path, Path]:
    manifest = _load_manifest(manifest_path)
    sample_tag = f"sample_{sample_id:06d}"
    symmetry = manifest.get("material_symmetry", "D6h")
    sym = resolve_symmetry(symmetry)
    sym_quats = mph.configure_symmetry(symmetry)
    ops = mph.conjugated_ops(sym_quats)

    run_by_name = {run["name"]: run for run in manifest["runs"]}
    ref_run = run_by_name.get("OCRP-direct-Reynolds-isometric-L6", manifest["runs"][0])
    ref_dir = _sr_dir(ref_run)
    lr = _load_q(ref_dir / f"{sample_tag}_lr.npy")
    hr = _load_q(ref_dir / f"{sample_tag}_hr.npy")
    out_hw = tuple(int(x) for x in hr.shape[:2])

    panels: list[tuple[str, np.ndarray, float | None]] = []
    sources: dict[str, str] = {}
    lr_rgb = _nearest_to_hw(_ipfz(lr, sym), out_hw)
    panels.append(("LR input", lr_rgb, None))
    sources["LR input"] = str(ref_dir / f"{sample_tag}_lr.npy")

    for method in CLASSICAL:
        sr = mph.method_field(method, None, lr, out_hw, sample_id)
        panels.append((method, _ipfz(sr, sym), _mean_deg(sr, hr, ops)))
        sources[method] = "computed from LR input"

    for method in LEARNED_ORDER:
        run = run_by_name[method]
        method_dir = _sr_dir(run)
        sr = mph.method_field(method, method_dir, lr, out_hw, sample_id)
        panels.append((DISPLAY_NAME.get(method, method), _ipfz(sr, sym), _mean_deg(sr, hr, ops)))
        sources[DISPLAY_NAME.get(method, method)] = str(method_dir / f"{sample_tag}_sr.npy")

    panels.append(("HR target", _ipfz(hr, sym), None))
    sources["HR target"] = str(ref_dir / f"{sample_tag}_hr.npy")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{sample_tag}_ipfz_all_methods.png"
    meta_path = out_dir / f"{sample_tag}_ipfz_all_methods.json"

    fig = plt.figure(figsize=(14.2, 4.7), dpi=dpi)
    gs = fig.add_gridspec(2, 8, width_ratios=[1] * 7 + [0.9], wspace=0.04, hspace=0.24)

    for idx, (label, rgb, mean_deg) in enumerate(panels):
        row, col = divmod(idx, 7)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(rgb, interpolation="nearest", resample=False)
        ax.set_aspect("equal")
        ax.axis("off")
        title = label if mean_deg is None else f"{label}\n{mean_deg:.2f} deg"
        weight = "bold" if label in {"OCRP", "HR target"} else "normal"
        ax.set_title(title, fontsize=8.0, fontweight=weight, pad=3.2, linespacing=0.95)

    add_ipf_key_panel(
        fig,
        gs[:, -1],
        sym,
        title=f"IPF-Z key\n{symmetry}",
        title_fontsize=8.0,
        label_fontsize=7.0,
    )
    fig.suptitle(
        f"Ti64 DIC McLean fresh 4x4, Test {sample_tag}: IPF-Z comparison",
        fontsize=10.5,
        y=0.995,
    )
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    meta = {
        "manifest": str(manifest_path),
        "sample_id": int(sample_id),
        "sample_tag": sample_tag,
        "symmetry": symmetry,
        "figure": str(png_path),
        "panel_order": [label for label, _, _ in panels],
        "panel_mean_misorientation_deg": {
            label: value for label, _, value in panels if value is not None
        },
        "sources": sources,
        "lr_shape": list(lr.shape),
        "hr_shape": list(hr.shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return png_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4" / "manifest.json",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4" / "visuals",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    png, meta = render(args.manifest, args.sample_id, args.out_dir, args.dpi)
    print(f"Wrote {png}")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
