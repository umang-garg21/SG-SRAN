#!/usr/bin/env python3
# %% [markdown]
"""
Full test-dataset comparison for all SR / interpolation methods on IN718_QSR_4x1.

This keeps the same overall format as `patch_sample_comparison_all_methods.ipynb`:
1. setup
2. metric helpers
3. model + dataset load
4. methods registry
5. full-test evaluation harness
6. run
7. representative full-map IPF overview
8. representative patch dashboard
9. full test-split scalar summary table

Important note on the final table:
- The table is computed from the whole test dataset, not from per-sample scalar means.
- For misorientation metrics, PSNR, and SSIM, we aggregate the full-resolution
  misorientation maps over the entire split.
- For GROD and KAM, we first compute each sample's scalar map independently, then
  concatenate those scalar maps across the whole split. This avoids invalid spatial
  neighborhoods across sample boundaries while still giving a whole-dataset result.
- The summary rows `mean|GROD - HR|` and `mean|KAM - HR|` are computed directly on
  those concatenated scalar maps, so they reflect whole-split scalar-map error
  instead of only the difference between two dataset means.
- Dataset-level SSIM over disconnected images is not uniquely defined. Here it is
  computed on the vertically concatenated misorientation map for the full split so
  it remains a whole-dataset metric rather than a per-sample average.

Plots and summary tables are saved to an analysis output folder. They are not
displayed interactively.
"""

# %% Section 1: Setup
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

repo_root = Path.cwd()
if not (repo_root / "training").exists():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "training").exists():
            repo_root = p
            break
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

analysis_dir = str(repo_root / "analysis")
if analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

from training.config_utils import load_and_prepare_config
from inference import infer_iso_embedding_sr_attn as inf
from models.bicubic_f_interpolate_sr import QuaternionBicubicFInterpolateSR
from utils import reduce_to_fz_min_angle
from utils.symmetry_utils import canon_symmetry_str, resolve_symmetry
from visualization.ipf_render import render_ipf_rgb

from slerp_final import (
    qnorm as _qnorm_t,
    slerp as _slerp_t,
    symmetrize_pair,
    bilinear_slerp_sym,
    make_fcc_symmetry_4x4,
)
from segment_grains import (
    segment_grains_graph,
    cleanup_small_grains_cuda,
    compute_gb_mask as _compute_gb_mask_t,
)

try:
    from skimage.metrics import structural_similarity as sk_ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    sk_ssim = None
    SKIMAGE_AVAILABLE = False

DATASET_DIR = Path("/data/home/umang/Materials/Materials_data_mount/datasets/IN718_QSR_4x1")
SPLIT = "Test"
TAKE_FIRST = None          # set an int for a smoke test
REPRESENTATIVE_INDEX = 0   # first sorted test sample by default
PATCH_ROWS = slice(50, 120)
PATCH_COLS = slice(50, 150)
SHOW_PLOTS = True
DISPLAY_PLOTS = False
SAVE_PLOTS = True
CONFIG_BATCH_SIZE = 1
GPU_ENV_KEY = "SR_ANALYSIS_GPU"
ARTIFACT_ROOT = repo_root / "analysis" / "_dataset_comparison_tmp"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = repo_root / "analysis" / "test_dataset_comparison_outputs" / f"{SPLIT.lower()}_{RUN_STAMP}"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESUME_CACHE_ROOT = repo_root / "analysis" / "test_dataset_comparison_resume" / DATASET_DIR.name / SPLIT.lower()
RESUME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
METRIC_DIR = OUTPUT_ROOT / "metrics"
for _path in (FIGURE_DIR, TABLE_DIR, METRIC_DIR):
    _path.mkdir(parents=True, exist_ok=True)


def _select_cuda_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), None

    visible_count = torch.cuda.device_count()
    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_physical = None
    if visible_spec is not None and str(visible_spec).strip() != "":
        try:
            visible_physical = [int(part.strip()) for part in str(visible_spec).split(",") if part.strip() != ""]
        except ValueError:
            visible_physical = None

    env_gpu = os.environ.get(GPU_ENV_KEY)
    if env_gpu is not None and str(env_gpu).strip() != "":
        gpu_idx = int(env_gpu)
        if 0 <= gpu_idx < visible_count:
            return torch.device(f"cuda:{gpu_idx}"), {
                "mode": "env_local",
                "gpu_index": gpu_idx,
                "visible_devices": visible_spec,
            }
        if visible_physical is not None and gpu_idx in visible_physical:
            local_idx = visible_physical.index(gpu_idx)
            return torch.device(f"cuda:{local_idx}"), {
                "mode": "env_physical_mapped",
                "gpu_index": gpu_idx,
                "local_index": local_idx,
                "visible_devices": visible_spec,
            }
        raise ValueError(
            f"{GPU_ENV_KEY}={gpu_idx} is not a valid visible CUDA device. "
            f"Visible local ordinals: 0..{visible_count - 1}; CUDA_VISIBLE_DEVICES={visible_spec!r}"
        )

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            idx_s, used_s, total_s = [part.strip() for part in line.split(",")]
            idx = int(idx_s)
            used = int(used_s)
            total = int(total_s)
            free = total - used
            rows.append({"gpu_index": idx, "used_mb": used, "total_mb": total, "free_mb": free})
        if visible_physical is not None:
            rows = [row for row in rows if row["gpu_index"] in visible_physical]
        if rows:
            best = max(rows, key=lambda r: (r["free_mb"], -r["used_mb"]))
            local_idx = (
                visible_physical.index(best["gpu_index"])
                if visible_physical is not None
                else best["gpu_index"]
            )
            return torch.device(f"cuda:{local_idx}"), {
                "mode": "auto_visible" if visible_physical is not None else "auto",
                "local_index": local_idx,
                "visible_devices": visible_spec,
                **best,
            }
    except Exception:
        pass

    return torch.device("cuda:0"), {
        "mode": "fallback",
        "gpu_index": 0,
        "visible_devices": visible_spec,
    }


device, device_info = _select_cuda_device()
if device.type == "cuda":
    torch.cuda.set_device(device)
model_load_device = torch.device("cpu") if torch.cuda.is_available() else device
sym = resolve_symmetry("Oh")

print("repo_root:", repo_root)
print("device:", device)
print("device_info:", device_info)
print("model_load_device:", model_load_device)
print("symmetry:", getattr(sym, "name", "Oh"))
print("dataset:", DATASET_DIR / SPLIT)
print("SKIMAGE_AVAILABLE:", SKIMAGE_AVAILABLE)
print("config_batch_size_override:", CONFIG_BATCH_SIZE)
print(f"{GPU_ENV_KEY}:", os.environ.get(GPU_ENV_KEY))
print("artifact_root:", ARTIFACT_ROOT)
print("output_root:", OUTPUT_ROOT)
print("resume_cache_root:", RESUME_CACHE_ROOT)


def _show_or_print(obj, fallback=None):
    if display is not None:
        display(obj)
    elif fallback is not None:
        print(fallback)
    else:
        print(obj)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path, payload, *, verbose=True):
    path = Path(path)
    with path.open("w") as f:
        json.dump(_to_builtin(payload), f, indent=2)
    if verbose:
        print("Saved JSON:", path)


def _read_json(path):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def _save_dataframe_bundle(df, csv_path, html_path, styler=None, float_format="%.6f"):
    csv_path = Path(csv_path)
    html_path = Path(html_path)
    df.to_csv(csv_path, float_format=float_format)
    if styler is None:
        html_path.write_text(df.to_html())
    else:
        html_path.write_text(styler.to_html())
    print("Saved table:", csv_path)
    print("Saved table:", html_path)


def _finalize_figure(fig, out_path, dpi=200):
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure:", out_path)
    if DISPLAY_PLOTS:
        plt.show()
    plt.close(fig)


def _safe_artifact_name(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def _slice_payload(slc):
    return {
        "start": slc.start,
        "stop": slc.stop,
        "step": slc.step,
    }


def _save_representative_cache(path, representative):
    path = Path(path)
    payload = {}
    for key, value in representative.items():
        if isinstance(value, str):
            payload[key] = np.asarray(value)
        else:
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def _load_representative_cache(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        representative = {}
        for key in data.files:
            value = data[key]
            representative[key] = value.item() if value.ndim == 0 else value
    return representative


def _evaluation_signature(method_name, pairs, representative_index, patch_rows, patch_cols, sym_ops):
    representative_sample_id = None
    if 0 <= representative_index < len(pairs):
        representative_sample_id = pairs[representative_index]["sample_id"]
    return {
        "resume_version": 1,
        "method_name": method_name,
        "dataset_dir": str(DATASET_DIR),
        "split": SPLIT,
        "take_first": TAKE_FIRST,
        "concat_shape": list(CONCAT_SHAPE),
        "n_samples": len(pairs),
        "sample_ids": [pair["sample_id"] for pair in pairs],
        "representative_index": int(representative_index),
        "representative_sample_id": representative_sample_id,
        "patch_rows": _slice_payload(patch_rows),
        "patch_cols": _slice_payload(patch_cols),
        "symmetry": _resolve_symmetry_name(sym_ops) or str(sym_ops),
    }


def _reset_resume_cache_dir(cache_dir):
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _prepare_resume_cache(method_name, pairs, representative_index, patch_rows, patch_cols, sym_ops):
    safe_method_name = _safe_artifact_name(method_name)
    cache_dir = RESUME_CACHE_ROOT / safe_method_name
    signature = _evaluation_signature(
        method_name,
        pairs,
        representative_index,
        patch_rows,
        patch_cols,
        sym_ops,
    )
    state_path = cache_dir / "state.json"
    representative_path = cache_dir / "representative.npz"
    final_result_path = cache_dir / "final_result.json"
    array_paths = {
        "mis": cache_dir / "mis.dat",
        "grod_sr": cache_dir / "grod_sr.dat",
        "grod_hr": cache_dir / "grod_hr.dat",
        "kam_sr": cache_dir / "kam_sr.dat",
        "kam_hr": cache_dir / "kam_hr.dat",
    }

    state = _read_json(state_path) if state_path.exists() else None
    arrays_exist = all(path.exists() for path in array_paths.values())
    cache_valid = (
        state is not None
        and state.get("signature") == signature
        and (
            arrays_exist
            or int(state.get("completed_samples", 0)) == 0
        )
    )
    if not cache_valid:
        _reset_resume_cache_dir(cache_dir)
        state = {
            "signature": signature,
            "status": "pending",
            "completed_samples": 0,
            "row_start": 0,
            "last_sample_id": None,
            "updated_at": datetime.now().isoformat(),
        }
        _write_json(state_path, state, verbose=False)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "cache_dir": cache_dir,
        "safe_method_name": safe_method_name,
        "signature": signature,
        "state_path": state_path,
        "representative_path": representative_path,
        "final_result_path": final_result_path,
        "array_paths": array_paths,
        "state": state,
    }


def _write_method_state(
    state_path,
    *,
    signature,
    status,
    completed_samples,
    row_start,
    last_sample_id,
):
    payload = {
        "signature": signature,
        "status": status,
        "completed_samples": int(completed_samples),
        "row_start": int(row_start),
        "last_sample_id": last_sample_id,
        "updated_at": datetime.now().isoformat(),
    }
    _write_json(state_path, payload, verbose=False)


def _progress_metrics_payload(results):
    return {
        name: {
            "global": results[name]["global"],
            "representative_sample_id": results[name]["representative"]["sample_id"],
        }
        for name in results
    }


def build_dataset_pairs(dataset_dir, split="Test", take_first=None):
    lr_dir = Path(dataset_dir) / split / "LR_Data"
    hr_dir = Path(dataset_dir) / split / "HR_Data"
    lr_files = sorted(lr_dir.glob("*.npy"))
    if take_first is not None:
        lr_files = lr_files[: int(take_first)]

    pairs = []
    for lr_file in lr_files:
        hr_name = lr_file.name.replace("_lr_", "_hr_")
        hr_file = hr_dir / hr_name
        if not hr_file.exists():
            print(f"WARNING: missing HR pair for {lr_file.name}, skipping.")
            continue
        pairs.append(
            {
                "sample_id": lr_file.stem.replace("_lr_x", ""),
                "lr_file": lr_file,
                "hr_file": hr_file,
            }
        )
    if not pairs:
        raise RuntimeError(f"No LR/HR pairs found under {dataset_dir / split}")
    return pairs


PAIRS = build_dataset_pairs(DATASET_DIR, split=SPLIT, take_first=TAKE_FIRST)
REP_PAIR = PAIRS[REPRESENTATIVE_INDEX]
print("n_samples:", len(PAIRS))
print("representative sample:", REP_PAIR["sample_id"])


def infer_concat_shape(pairs):
    total_rows = 0
    widths = set()
    for pair in pairs:
        hr_shape = np.load(pair["hr_file"], mmap_mode="r").shape
        if len(hr_shape) != 3 or hr_shape[-1] != 4:
            raise ValueError(f"Expected HR quaternion array of shape (H, W, 4), got {hr_shape}")
        total_rows += int(hr_shape[0])
        widths.add(int(hr_shape[1]))
    if len(widths) != 1:
        raise ValueError(
            f"Whole-dataset concatenation for SSIM expects a shared width, found widths: {sorted(widths)}"
        )
    return (total_rows, next(iter(widths)))


CONCAT_SHAPE = infer_concat_shape(PAIRS)
print("concat_shape_for_dataset_metrics:", CONCAT_SHAPE)

# %% Section 2: Quaternion + metric helpers
def normalize_quat(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0:
        return q
    if q.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have last axis size 4")
    n = np.linalg.norm(q.reshape(-1, 4), axis=1).reshape(q.shape[:-1] + (1,))
    return q / (n + eps)


def quat_conjugate(q):
    q = np.asarray(q)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("Input quaternions must have shape (..., 4)")
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack((w, x, y, z), axis=-1)


def crystallographic_misorientation(q_pred, q_gt, sym_quats=None, degrees=True):
    q_pred = np.asarray(q_pred, dtype=np.float64)
    q_gt = np.asarray(q_gt, dtype=np.float64)

    if q_pred.shape[-1] != 4 or q_gt.shape[-1] != 4:
        raise ValueError("Input quaternions must have last axis length 4")

    q_pred = normalize_quat(q_pred)
    q_gt = normalize_quat(q_gt)

    if sym_quats is None:
        sym_qs = None
    elif isinstance(sym_quats, str):
        sym_qs = np.asarray(resolve_symmetry(sym_quats).data, dtype=np.float64)
    elif hasattr(sym_quats, "data"):
        sym_qs = np.asarray(sym_quats.data, dtype=np.float64)
    elif hasattr(sym_quats, "to_quaternion"):
        sym_qs = np.asarray(sym_quats.to_quaternion(), dtype=np.float64)
    elif isinstance(sym_quats, np.ndarray):
        sym_qs = np.asarray(sym_quats, dtype=np.float64)
        if sym_qs.ndim != 2 or sym_qs.shape[1] != 4:
            raise ValueError("sym_quats numpy array must have shape (G, 4)")
    else:
        raise ValueError("Unknown sym_quats type")

    shape = q_pred.shape[:-1]
    qp = q_pred.reshape(-1, 4)
    qg = q_gt.reshape(-1, 4)

    if sym_qs is None:
        dots = np.abs(np.sum(qp * qg, axis=-1))
        dots = np.clip(dots, -1.0, 1.0)
        ang = 2.0 * np.arccos(dots)
        if degrees:
            ang = np.degrees(ang)
        return ang.reshape(shape)

    ops = sym_qs.copy()
    ops[:, 1:] *= -1.0
    gq = quat_mul(ops[:, None, :], qg[None, :, :])
    dots = np.abs(np.sum(gq * qp[None, :, :], axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    ang = 2.0 * np.arccos(dots)
    min_ang = np.min(ang, axis=0)
    if degrees:
        min_ang = np.degrees(min_ang)
    return min_ang.reshape(shape)


def misorientation_map(pred_q, gt_q, sym_quats=None, degrees=True):
    return crystallographic_misorientation(pred_q, gt_q, sym_quats=sym_quats, degrees=degrees)


def quaternion_mean(qs):
    qs = np.asarray(qs, dtype=np.float64).reshape(-1, 4)
    qs = normalize_quat(qs)
    M = np.dot(qs.T, qs)
    w, v = np.linalg.eigh(M)
    avg = v[:, np.argmax(w)]
    if avg[0] < 0:
        avg = -avg
    return normalize_quat(avg)


def compute_grod(ori_map, grain_labels=None, sym_ops=None, window_size=21):
    ori_map = np.asarray(ori_map, dtype=np.float64)
    H, W = ori_map.shape[:2]
    if grain_labels is not None:
        labels = np.asarray(grain_labels)
        grod = np.zeros((H, W), dtype=np.float64)
        grain_stats = {}
        for lab in np.unique(labels):
            mask = labels == lab
            if np.count_nonzero(mask) == 0:
                continue
            qs = ori_map[mask]
            mean_q = quaternion_mean(qs)
            grod_vals = crystallographic_misorientation(
                qs, np.tile(mean_q, (qs.shape[0], 1)), sym_quats=sym_ops, degrees=True
            )
            grod[mask] = grod_vals
            grain_stats[int(lab)] = {
                "mean": float(np.nanmean(grod_vals)),
                "max": float(np.nanmax(grod_vals)),
                "std": float(np.nanstd(grod_vals)),
                "count": int(qs.shape[0]),
            }
        return grod, grain_stats

    q = normalize_quat(ori_map)
    M_local = np.zeros((H, W, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i, 4):
            arr = q[..., i] * q[..., j]
            local = uniform_filter(arr, size=window_size, mode="reflect")
            M_local[..., i, j] = local
            M_local[..., j, i] = local
    M_flat = M_local.reshape(-1, 4, 4)
    _, v = np.linalg.eigh(M_flat)
    avg = v[:, :, -1]
    signs = np.where(avg[:, 0] < 0, -1.0, 1.0).reshape(-1, 1)
    avg = (avg * signs).reshape(H, W, 4)
    avg = normalize_quat(avg)
    grod = crystallographic_misorientation(ori_map, avg, sym_quats=sym_ops, degrees=True)
    stats = {
        "mean": float(np.nanmean(grod)),
        "max": float(np.nanmax(grod)),
        "std": float(np.nanstd(grod)),
    }
    return grod, stats


def compute_kam(
    ori_map,
    grain_labels=None,
    radius=1,
    sym_ops=None,
    ignore_threshold_deg=15.0,
):
    ori_map = np.asarray(ori_map, dtype=np.float64)
    H, W = ori_map.shape[:2]
    neighbors = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            neighbors.append((dy, dx))

    vals = np.zeros((len(neighbors), H, W), dtype=np.float64)
    for k, (dy, dx) in enumerate(neighbors):
        src_y0 = max(0, -dy)
        src_y1 = H - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = W - max(0, dx)
        dst_y0 = max(0, dy)
        dst_y1 = H - max(0, -dy)
        dst_x0 = max(0, dx)
        dst_x1 = W - max(0, -dx)

        shifted = np.zeros_like(ori_map)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = ori_map[src_y0:src_y1, src_x0:src_x1]

        valid_mask = np.zeros((H, W), dtype=bool)
        valid_mask[dst_y0:dst_y1, dst_x0:dst_x1] = True
        if grain_labels is not None:
            same_grain = np.zeros((H, W), dtype=bool)
            same_grain[dst_y0:dst_y1, dst_x0:dst_x1] = (
                grain_labels[dst_y0:dst_y1, dst_x0:dst_x1]
                == grain_labels[src_y0:src_y1, src_x0:src_x1]
            )
            valid_mask = valid_mask & same_grain

        mis = np.full((H, W), np.nan, dtype=np.float64)
        if np.any(valid_mask):
            mis_vals = crystallographic_misorientation(
                ori_map[valid_mask], shifted[valid_mask], sym_quats=sym_ops, degrees=True
            )
            mis[valid_mask] = mis_vals
        if ignore_threshold_deg is not None:
            mis[mis > ignore_threshold_deg] = np.nan
        vals[k] = mis

    valid = np.isfinite(vals)
    counts = valid.sum(axis=0)
    kam = np.full((H, W), np.nan, dtype=np.float64)
    if np.any(counts > 0):
        summed = np.where(valid, vals, 0.0).sum(axis=0)
        kam[counts > 0] = summed[counts > 0] / counts[counts > 0]
    stats = {
        "mean": float(np.nanmean(kam)),
        "max": float(np.nanmax(kam)),
        "std": float(np.nanstd(kam)),
    }
    return kam, stats


def _resolve_symmetry_quats(sym_quats):
    if sym_quats is None:
        return None
    if isinstance(sym_quats, str):
        return np.asarray(resolve_symmetry(sym_quats).data, dtype=np.float64)
    if hasattr(sym_quats, "data"):
        return np.asarray(sym_quats.data, dtype=np.float64)
    if hasattr(sym_quats, "to_quaternion"):
        return np.asarray(sym_quats.to_quaternion(), dtype=np.float64)
    arr = np.asarray(sym_quats, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("sym_quats numpy array must have shape (G, 4)")
    return arr


KNOWN_MAX_MISORIENTATION_DEG = {
    "C1": 180.0,
    "Ci": 180.0,
    "C2h": 180.0,
    "D2h": 180.0,
    "D4h": 90.0,
    "D6h": 93.84,
    "Oh": 62.8,
}


def _resolve_symmetry_name(sym_quats):
    if sym_quats is None:
        return None
    if isinstance(sym_quats, str):
        return canon_symmetry_str(sym_quats)
    for attr in ("name", "__name__"):
        value = getattr(sym_quats, attr, None)
        if value:
            return canon_symmetry_str(str(value))
    return None


def max_misorientation_from_sym(sym_quats):
    # The pairwise angle between symmetry operators is not the same as the
    # maximum crystallographic disorientation inside the fundamental zone.
    # For PSNR/SSIM we want the latter so the dynamic range stays physically meaningful.
    sym_name = _resolve_symmetry_name(sym_quats)
    if sym_name in KNOWN_MAX_MISORIENTATION_DEG:
        return float(KNOWN_MAX_MISORIENTATION_DEG[sym_name])
    return 180.0


def _safe_nanpercentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _summarize_scalar_map_pair(sr_concat, hr_concat, *, percentile_key, percentile):
    sr_concat = np.asarray(sr_concat, dtype=np.float64)
    hr_concat = np.asarray(hr_concat, dtype=np.float64)
    delta = sr_concat - hr_concat
    abs_delta = np.abs(delta)

    sr_mean = float(np.nanmean(sr_concat))
    hr_mean = float(np.nanmean(hr_concat))

    return {
        "mean": sr_mean,
        "std": float(np.nanstd(sr_concat)),
        percentile_key: _safe_nanpercentile(sr_concat, percentile),
        "hr_mean": hr_mean,
        "hr_std": float(np.nanstd(hr_concat)),
        f"hr_{percentile_key}": _safe_nanpercentile(hr_concat, percentile),
        "mean_diff": float(sr_mean - hr_mean),
        "abs_mean_diff": float(abs(sr_mean - hr_mean)),
        "mean_abs_delta": float(np.nanmean(abs_delta)),
        "rmse_delta": float(np.sqrt(np.nanmean(delta ** 2))),
    }


def psnr_from_map(ref_map, test_map, max_val, eps=1e-10):
    ref_map = np.asarray(ref_map, dtype=np.float64)
    test_map = np.asarray(test_map, dtype=np.float64)
    mse = np.nanmean((ref_map - test_map) ** 2)
    if not np.isfinite(mse) or mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(float(max_val) / np.sqrt(mse + eps)))


def ssim_from_map(ref_map, test_map, win_size=7, data_range=None):
    if not SKIMAGE_AVAILABLE or sk_ssim is None:
        return np.nan
    ref_map = np.asarray(ref_map, dtype=np.float64)
    test_map = np.asarray(test_map, dtype=np.float64)
    if data_range is None:
        data_range = float(
            np.nanmax([np.nanmax(ref_map), np.nanmax(test_map)])
            - np.nanmin([np.nanmin(ref_map), np.nanmin(test_map)])
        )
    if not np.isfinite(data_range) or data_range <= 0:
        return 1.0
    win_size = min(int(win_size), ref_map.shape[0], ref_map.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)
    return float(
        sk_ssim(ref_map, test_map, data_range=float(data_range), win_size=win_size, channel_axis=None)
    )


# %% Section 3: Load models + representative test sample
rep_lr = np.load(REP_PAIR["lr_file"]).astype(np.float32)
rep_hr = np.load(REP_PAIR["hr_file"]).astype(np.float32)
rep_hr_patch = rep_hr[PATCH_ROWS, PATCH_COLS]
rep_grod_hr_patch, _ = compute_grod(rep_hr_patch, sym_ops=sym)
rep_kam_hr_patch, _ = compute_kam(rep_hr_patch, radius=1, sym_ops=sym)
sample_scale = (int(rep_hr.shape[0] // rep_lr.shape[0]), int(rep_hr.shape[1] // rep_lr.shape[1]))


def _as_scale_tuple(scale_value):
    if isinstance(scale_value, (list, tuple)):
        return (int(scale_value[0]), int(scale_value[1]))
    scale_value = int(scale_value)
    return (scale_value, scale_value)


def _resolve_run_cfg_path(exp_dir):
    candidates = [
        exp_dir / "logs" / "inference_run_config.json",
        exp_dir / "logs" / "run_config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No run config found under {exp_dir / 'logs'}")


def _resolve_checkpoint_path(cfg_local, exp_dir, checkpoint_name="best_model.pt", checkpoint_dir=None):
    if checkpoint_name is None:
        raise ValueError("checkpoint_name must be provided when using explicit checkpoint folder resolution.")
    ckpt_candidate = Path(checkpoint_name)
    if ckpt_candidate.is_absolute():
        return ckpt_candidate
    checkpoints_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else Path(getattr(cfg_local, "checkpoints_dir", exp_dir / "checkpoints"))
    )
    return checkpoints_dir / checkpoint_name


def _load_experiment_model(
    exp_dir,
    *,
    override_scale=None,
    checkpoint_path=None,
    checkpoint_name="best_model.pt",
    checkpoint_dir=None,
    load_device=model_load_device,
):
    cfg_path = exp_dir / "config_new.json"
    run_cfg_path = _resolve_run_cfg_path(exp_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config missing: {cfg_path}")
    cfg_local = load_and_prepare_config(cfg_path, run_cfg_path)
    cfg_local.batch_size = int(CONFIG_BATCH_SIZE)
    if override_scale is not None:
        override_scale = _as_scale_tuple(override_scale)
        original_scale = _as_scale_tuple(
            getattr(cfg_local, "upsample_factor", getattr(cfg_local, "scale", override_scale))
        )
        if original_scale != override_scale:
            print(f"Overriding {exp_dir.name} scale from {original_scale} to {override_scale} for this dataset.")
        cfg_local.scale = list(override_scale)
        cfg_local.upsample_factor = list(override_scale)
    if checkpoint_path is None:
        checkpoint = _resolve_checkpoint_path(
            cfg_local,
            exp_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    model_local = inf._load_model_from_checkpoint(cfg_local, checkpoint, load_device)
    model_local.eval()
    return cfg_local, model_local, checkpoint


base_exp_dir = repo_root / "experiments/IN718/iso_embedding_sr_attn_01"
boundary_aware_exp_dir = repo_root / "experiments/IN718/iso_embedding_sr_attn_boundary_aware_01"
boundary_aware_checkpoint_dir = boundary_aware_exp_dir / "checkpoints"
sdf_one_sided_exp_dir = repo_root / "experiments/IN718/irrep_sdf_sr_one_sided_01"
qrbsa_exp_dir = repo_root / "experiments/IN718/qrbsa_01"
cfg = None
model = None
checkpoint = None
cfg_boundary_aware = None
model_boundary_aware = None
checkpoint_boundary_aware = None
cfg_sdf_one_sided = None
model_sdf_one_sided = None
checkpoint_sdf_one_sided = None
model_qrbsa = None


def _unload_base_model():
    global model
    if model is not None:
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass
        del model
        model = None


def _unload_boundary_aware_model():
    global model_boundary_aware
    if model_boundary_aware is not None:
        try:
            model_boundary_aware.to(torch.device("cpu"))
        except Exception:
            pass
        del model_boundary_aware
        model_boundary_aware = None


def _ensure_base_model_loaded():
    global cfg, model, checkpoint
    if model is None:
        cfg, model, checkpoint = _load_experiment_model(base_exp_dir, override_scale=sample_scale)
        print("Base model loaded:", type(model).__name__, "from", checkpoint)
    return model


def _ensure_boundary_aware_model_loaded():
    global cfg_boundary_aware, model_boundary_aware, checkpoint_boundary_aware
    if model_boundary_aware is None:
        cfg_boundary_aware, model_boundary_aware, checkpoint_boundary_aware = _load_experiment_model(
            boundary_aware_exp_dir,
            override_scale=sample_scale,
            checkpoint_dir=boundary_aware_checkpoint_dir,
        )
        print("Boundary-aware model loaded:", type(model_boundary_aware).__name__, "from", checkpoint_boundary_aware)
    return model_boundary_aware


def _unload_sdf_one_sided_model():
    global model_sdf_one_sided
    if model_sdf_one_sided is not None:
        try:
            model_sdf_one_sided.to(torch.device("cpu"))
        except Exception:
            pass
        del model_sdf_one_sided
        model_sdf_one_sided = None


def _ensure_sdf_one_sided_model_loaded():
    global cfg_sdf_one_sided, model_sdf_one_sided, checkpoint_sdf_one_sided
    if model_sdf_one_sided is None:
        cfg_sdf_one_sided, model_sdf_one_sided, checkpoint_sdf_one_sided = _load_experiment_model(
            sdf_one_sided_exp_dir,
            override_scale=sample_scale,
        )
        print("SDF 1sided model loaded:", type(model_sdf_one_sided).__name__, "from", checkpoint_sdf_one_sided)
    return model_sdf_one_sided


def _unload_qrbsa_model():
    global model_qrbsa
    if model_qrbsa is not None:
        try:
            model_qrbsa.to(torch.device("cpu"))
        except Exception:
            pass
        del model_qrbsa
        model_qrbsa = None


def _ensure_qrbsa_model_loaded():
    global model_qrbsa
    if model_qrbsa is None:
        _qrbsa_root = repo_root / "Q-RBSA"
        for _p in [str(_qrbsa_root), str(_qrbsa_root / "model")]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from qrbsa_1d import QRBSA_1D  # noqa: E402
        from types import SimpleNamespace as _SN
        with open(qrbsa_exp_dir / "config.json") as _f:
            _qrbsa_cfg = json.load(_f)
        model_qrbsa = QRBSA_1D(_SN(
            n_colors=4,
            n_resblocks=_qrbsa_cfg.get("n_resblocks", 16),
            n_feats=_qrbsa_cfg.get("n_feats", 64),
            scale=_qrbsa_cfg.get("scale", 4),
        )).to(model_load_device)
        _ckpt = torch.load(qrbsa_exp_dir / "checkpoints" / "best_model.pt", map_location=model_load_device)
        model_qrbsa.load_state_dict(_ckpt["model_state_dict"])
        model_qrbsa.eval()
        print("QRBSA model loaded:", type(model_qrbsa).__name__, "from", qrbsa_exp_dir / "checkpoints/best_model.pt")
    return model_qrbsa


print("Representative LR shape:", rep_lr.shape)
print("Representative HR shape:", rep_hr.shape)
print("Representative patch shape:", rep_hr_patch.shape)
print("Sample scale:", sample_scale)

model_bicubic_finterp = QuaternionBicubicFInterpolateSR(
    upsample_factor=sample_scale,
    align_corners=False,
    normalize_output=True,
    canonicalize_output=False,
    device=device,
).eval()

# %% Section 4: Methods registry
def _hwc4(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 array, got {arr.shape}")
    if arr.shape[-1] == 4:
        return arr
    if arr.shape[0] == 4:
        return np.moveaxis(arr, 0, -1)
    raise ValueError(f"No axis of size 4 found in {arr.shape}")


def _normalize_quat_torch(quats, eps=1e-12):
    quats = quats / torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(quats[..., :1] < 0.0, -quats, quats)


def _forward_model_sr_split(model_obj, lr_flat, lr_shape, *, lr_boundary_map=None):
    with torch.no_grad():
        lr_flat = _normalize_quat_torch(lr_flat)
        feat_lr_a1 = model_obj.encode_a1(lr_flat)
        if lr_boundary_map is None:
            feat_hr_a1, _ = model_obj._forward_sr_features(feat_lr_a1, lr_shape)
        else:
            feat_hr_a1, _ = model_obj._forward_sr_features(
                feat_lr_a1,
                lr_shape,
                lr_boundary_map=lr_boundary_map,
            )
        feat_hr_a1 = feat_hr_a1.detach()

    with torch.enable_grad():
        sr_flat = model_obj.decode(feat_hr_a1)
    return sr_flat


def upsample_model_sr(lr_q, out_hw):
    _ensure_base_model_loaded()
    lr_q = _hwc4(lr_q)
    lr_t = torch.from_numpy(lr_q).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = inf._flatten_quat_chw(lr_t)
    sr_flat = _forward_model_sr_split(model, lr_flat, lr_shape)
    sr_np = sr_flat.detach().cpu().numpy().reshape(int(out_hw[0]), int(out_hw[1]), 4)
    del lr_t, lr_flat, sr_flat
    return sr_np


def _compute_lr_boundary_map_torch(lr_q, angle_deg=5.0, mark_both_sides=True):
    q = torch.from_numpy(_hwc4(lr_q)).to(dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    h, w, _ = q.shape
    boundary = torch.zeros((h, w), dtype=torch.float32)
    theta = max(0.0, min(180.0, float(angle_deg)))
    cos_half = float(np.cos(np.deg2rad(0.5 * theta)))

    dot_x = (q[:, 1:, :] * q[:, :-1, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
    dot_y = (q[1:, :, :] * q[:-1, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
    b_x = (dot_x < cos_half).to(dtype=torch.float32)
    b_y = (dot_y < cos_half).to(dtype=torch.float32)

    if mark_both_sides:
        boundary[:, 1:] = torch.maximum(boundary[:, 1:], b_x)
        boundary[:, :-1] = torch.maximum(boundary[:, :-1], b_x)
        boundary[1:, :] = torch.maximum(boundary[1:, :], b_y)
        boundary[:-1, :] = torch.maximum(boundary[:-1, :], b_y)
    else:
        boundary[:, 1:] = torch.maximum(boundary[:, 1:], b_x)
        boundary[1:, :] = torch.maximum(boundary[1:, :], b_y)
    return boundary


def upsample_boundary_aware_model_sr(lr_q, out_hw):
    _ensure_boundary_aware_model_loaded()
    lr_q = _hwc4(lr_q)
    lr_t = torch.from_numpy(lr_q).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = inf._flatten_quat_chw(lr_t)
    lr_boundary = _compute_lr_boundary_map_torch(
        lr_q,
        angle_deg=float(getattr(cfg_boundary_aware, "lr_boundary_angle_deg", 5.0)),
        mark_both_sides=bool(getattr(cfg_boundary_aware, "lr_boundary_mark_both_sides", True)),
    ).to(device=device, dtype=torch.float32)
    sr_flat = _forward_model_sr_split(
        model_boundary_aware,
        lr_flat,
        lr_shape,
        lr_boundary_map=lr_boundary,
    )
    sr_np = sr_flat.detach().cpu().numpy().reshape(int(out_hw[0]), int(out_hw[1]), 4)
    if sr_np.shape[:2] != (int(out_hw[0]), int(out_hw[1])):
        raise ValueError(
            f"Boundary-aware SR shape mismatch: got {sr_np.shape[:2]}, expected {(int(out_hw[0]), int(out_hw[1]))}"
        )
    del lr_t, lr_flat, lr_boundary, sr_flat
    return sr_np


def upsample_sdf_one_sided_sr(lr_q, out_hw):
    _ensure_sdf_one_sided_model_loaded()
    lr_q = _hwc4(lr_q)
    lr_t = torch.from_numpy(lr_q).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = inf._flatten_quat_chw(lr_t)
    sr_flat = _forward_model_sr_split(model_sdf_one_sided, lr_flat, lr_shape)
    sr_np = sr_flat.detach().cpu().numpy().reshape(int(out_hw[0]), int(out_hw[1]), 4)
    del lr_t, lr_flat, sr_flat
    return sr_np


def upsample_qrbsa_sr(lr_q, out_hw):
    _ensure_qrbsa_model_loaded()
    lr_q = _hwc4(lr_q).astype(np.float32, copy=False)  # passive, scalar-first, (H, W, 4)
    lr_active = quat_conjugate(lr_q).astype(np.float32, copy=False)
    lr_t = torch.from_numpy(lr_active).to(device=device, dtype=torch.float32)
    lr_t = lr_t.permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W), active scalar-first
    lr_t = lr_t[:, [1, 2, 3, 0], :, :]  # active scalar-last for QRBSA
    with torch.no_grad():
        sr_t = model_qrbsa(lr_t)  # (1, 4, 4H, W), active scalar-last
    sr_t = sr_t[:, [3, 0, 1, 2], :, :]  # back to active scalar-first
    sr_t = F.normalize(sr_t, p=2, dim=1)
    sr_active = sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
    sr_passive = quat_conjugate(sr_active).astype(np.float32, copy=False)
    sr_np = reduce_to_fz_min_angle(
        sr_passive,
        sym=sym,
        normalize=True,
        hemisphere=True,
        return_op_map=False,
    ).astype(np.float32, copy=False)
    if sr_np.shape[:2] != (int(out_hw[0]), int(out_hw[1])):
        raise ValueError(
            f"QRBSA SR shape mismatch: got {sr_np.shape[:2]}, expected {(int(out_hw[0]), int(out_hw[1]))}"
        )
    del lr_t, sr_t
    return sr_np


def upsample_bicubic(lr_q, out_hw):
    lr_q = _hwc4(lr_q)
    h_out, w_out = int(out_hw[0]), int(out_hw[1])
    sr = np.zeros((h_out, w_out, 4), dtype=np.float32)
    for i in range(4):
        sr[..., i] = cv2.resize(lr_q[..., i], (w_out, h_out), interpolation=cv2.INTER_CUBIC)
    norms = np.linalg.norm(sr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return sr / norms


def upsample_bicubic_finterp(lr_q, out_hw):
    lr_q = _hwc4(lr_q)
    lr_t = torch.from_numpy(lr_q).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = inf._flatten_quat_chw(lr_t)
    with torch.no_grad():
        sr_flat = model_bicubic_finterp.forward_sr(
            lr_flat,
            lr_shape=lr_shape,
            normalize_input=False,
        )
    sr_np = sr_flat.detach().cpu().numpy().reshape(int(out_hw[0]), int(out_hw[1]), 4)
    if sr_np.shape[:2] != (int(out_hw[0]), int(out_hw[1])):
        raise ValueError(
            f"Torch bicubic SR shape mismatch: got {sr_np.shape[:2]}, expected {(int(out_hw[0]), int(out_hw[1]))}"
        )
    del lr_t, lr_flat, sr_flat
    return sr_np.astype(np.float32, copy=False)


def upsample_iso_bicubic(lr_q, out_hw):
    _ensure_base_model_loaded()
    lr_q = _hwc4(lr_q)
    h_lr, w_lr = lr_q.shape[:2]
    h_out, w_out = int(out_hw[0]), int(out_hw[1])

    lr_t = torch.from_numpy(lr_q.reshape(-1, 4)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feat_lr = model.encode_a1(lr_t)
    c = feat_lr.shape[-1]
    feat_lr_hwc = feat_lr.cpu().numpy().reshape(h_lr, w_lr, c)

    feat_hr_hwc = np.zeros((h_out, w_out, c), dtype=np.float32)
    for i in range(c):
        feat_hr_hwc[..., i] = cv2.resize(
            feat_lr_hwc[..., i], (w_out, h_out), interpolation=cv2.INTER_CUBIC
        )

    feat_hr_flat = torch.from_numpy(feat_hr_hwc.reshape(h_out * w_out, c)).to(
        device=device, dtype=torch.float32
    )
    with torch.enable_grad():
        sr_t = model.decode(feat_hr_flat)
    sr_np = sr_t.detach().cpu().numpy().reshape(h_out, w_out, 4)
    del lr_t, feat_lr, feat_hr_flat, sr_t
    return sr_np


def upsample_nn(lr_q, out_hw):
    lr_q = _hwc4(lr_q)
    h_out, w_out = int(out_hw[0]), int(out_hw[1])
    sr = np.zeros((h_out, w_out, 4), dtype=np.float32)
    for i in range(4):
        sr[..., i] = cv2.resize(lr_q[..., i], (w_out, h_out), interpolation=cv2.INTER_NEAREST)
    norms = np.linalg.norm(sr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return sr / norms


def upsample_iso_nn(lr_q, out_hw):
    _ensure_base_model_loaded()
    lr_q = _hwc4(lr_q)
    h_lr, w_lr = lr_q.shape[:2]
    h_out, w_out = int(out_hw[0]), int(out_hw[1])

    lr_t = torch.from_numpy(lr_q.reshape(-1, 4)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feat_lr = model.encode_a1(lr_t)
    c = feat_lr.shape[-1]
    feat_lr_hwc = feat_lr.cpu().numpy().reshape(h_lr, w_lr, c)

    feat_hr_hwc = np.zeros((h_out, w_out, c), dtype=np.float32)
    for i in range(c):
        feat_hr_hwc[..., i] = cv2.resize(
            feat_lr_hwc[..., i], (w_out, h_out), interpolation=cv2.INTER_NEAREST
        )

    feat_hr_flat = torch.from_numpy(feat_hr_hwc.reshape(h_out * w_out, c)).to(
        device=device, dtype=torch.float32
    )
    with torch.enable_grad():
        sr_t = model.decode(feat_hr_flat)
    sr_np = sr_t.detach().cpu().numpy().reshape(h_out, w_out, 4)
    del lr_t, feat_lr, feat_hr_flat, sr_t
    return sr_np


_interp_dev = torch.device("cpu")
_slerp_sym_ops_4x4 = make_fcc_symmetry_4x4(device="cpu", dtype=torch.float32)


def _slerp_upsample(lr_q, out_hw, sym_ops_4x4=None, dev=_interp_dev):
    lr_q = _hwc4(lr_q)
    h_out, w_out = int(out_hw[0]), int(out_hw[1])

    x = _qnorm_t(torch.from_numpy(lr_q).permute(2, 0, 1).unsqueeze(0).to(dev, torch.float32))
    B, _, H, W = x.shape

    iy = torch.arange(h_out, device=dev, dtype=x.dtype)
    ix = torch.arange(w_out, device=dev, dtype=x.dtype)
    y = (iy + 0.5) / (h_out / H) - 0.5
    xcoord = (ix + 0.5) / (w_out / W) - 0.5
    y0 = torch.floor(y).clamp(0, H - 1).long()
    x0 = torch.floor(xcoord).clamp(0, W - 1).long()
    y1 = (y0 + 1).clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    v = (y - y0.to(x.dtype)).clamp(0, 1)
    u = (xcoord - x0.to(x.dtype)).clamp(0, 1)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    y0g = y0.view(h_out, 1).expand(h_out, w_out)
    y1g = y1.view(h_out, 1).expand(h_out, w_out)
    x0g = x0.view(1, w_out).expand(h_out, w_out)
    x1g = x1.view(1, w_out).expand(h_out, w_out)

    def _flat(q):
        return q.permute(0, 2, 3, 1).reshape(-1, 4)

    q00_f = _flat(x[:, :, y0g, x0g])
    q01_f = _flat(x[:, :, y0g, x1g])
    q10_f = _flat(x[:, :, y1g, x0g])
    q11_f = _flat(x[:, :, y1g, x1g])

    u_full = u_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)
    v_full = v_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)

    if sym_ops_4x4 is None:
        q0u = _slerp_t(q00_f, q01_f, u_full)
        q1u = _slerp_t(q10_f, q11_f, u_full)
        q_uv = _slerp_t(q0u, q1u, v_full)
    else:
        sym_ops_4x4 = sym_ops_4x4.to(device=dev, dtype=x.dtype)
        q00m, q01m = symmetrize_pair(q00_f, q01_f, sym_ops_4x4)
        q0u = _slerp_t(q00m, q01m, u_full)
        q10m, q11m = symmetrize_pair(q10_f, q11_f, sym_ops_4x4)
        q1u = _slerp_t(q10m, q11m, u_full)
        q0u_m, q1u_m = symmetrize_pair(q0u, q1u, sym_ops_4x4)
        q_uv = _slerp_t(q0u_m, q1u_m, v_full)

    return _qnorm_t(q_uv).view(B, h_out, w_out, 4)[0].cpu().numpy().astype(np.float32)


def upsample_slerp(lr_q, out_hw):
    return _slerp_upsample(lr_q, out_hw, sym_ops_4x4=None, dev=_interp_dev)


def upsample_symm_slerp(lr_q, out_hw):
    return _slerp_upsample(lr_q, out_hw, sym_ops_4x4=_slerp_sym_ops_4x4, dev=_interp_dev)


_ba_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ba_sym_ops_cpu = make_fcc_symmetry_4x4(device="cpu", dtype=torch.float32)


@torch.no_grad()
def _build_smooth_hr_labels_aniso(labels_lr, scale_y, scale_x=1, sdf_shift=0.7):
    dev = labels_lr.device
    H, W = labels_lr.shape
    H_hr, W_hr = H * scale_y, W * scale_x

    gb_lr = _compute_gb_mask_t(labels_lr)
    gb_hr = F.interpolate(gb_lr[None, None], size=(H_hr, W_hr), mode="nearest")[0, 0]

    kernel = torch.ones((1, 1, 3, 3), device=dev) / 9.0
    dist = gb_hr.clone()
    for _ in range(12):
        dist = F.conv2d(dist[None, None], kernel, padding=1)[0, 0]

    sigma = 2.0
    ks = int(6 * sigma + 1) | 1
    half = ks // 2
    coords = torch.arange(ks, device=dev, dtype=torch.float32) - half
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)[None, None]
    sdf_hr = F.conv2d(dist[None, None], g2, padding=half)[0, 0]
    sdf_hr = sdf_hr / (sdf_hr.max() + 1e-8)

    dy = torch.zeros_like(sdf_hr)
    dx = torch.zeros_like(sdf_hr)
    dy[1:-1, :] = 0.5 * (sdf_hr[2:, :] - sdf_hr[:-2, :])
    dy[0, :] = sdf_hr[1, :] - sdf_hr[0, :]
    dy[-1, :] = sdf_hr[-1, :] - sdf_hr[-2, :]
    dx[:, 1:-1] = 0.5 * (sdf_hr[:, 2:] - sdf_hr[:, :-2])
    dx[:, 0] = sdf_hr[:, 1] - sdf_hr[:, 0]
    dx[:, -1] = sdf_hr[:, -1] - sdf_hr[:, -2]
    norm = torch.sqrt(dx * dx + dy * dy + 1e-12)
    nx, ny = dx / norm, dy / norm

    idx_y = torch.arange(H_hr, device=dev, dtype=torch.float32)
    idx_x = torch.arange(W_hr, device=dev, dtype=torch.float32)
    y_base = (idx_y + 0.5) / scale_y - 0.5
    x_base = (idx_x + 0.5) / scale_x - 0.5
    y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing="ij")

    y_nn = torch.round(y_grid - sdf_shift * ny).clamp(0, H - 1).long()
    x_nn = torch.round(x_grid - sdf_shift * nx).clamp(0, W - 1).long()
    return labels_lr[y_nn, x_nn].long()


@torch.no_grad()
def _analyze_boundary_cases_aniso(labels_lr, labels_hr, scale_y, scale_x=1):
    dev = labels_lr.device
    H_lr, W_lr = labels_lr.shape
    H_hr, W_hr = labels_hr.shape

    yy = torch.arange(H_hr, device=dev, dtype=torch.float32)
    xx = torch.arange(W_hr, device=dev, dtype=torch.float32)
    y_hr_g, x_hr_g = torch.meshgrid(yy, xx, indexing="ij")

    y_lr_f = (y_hr_g + 0.5) / scale_y - 0.5
    x_lr_f = (x_hr_g + 0.5) / scale_x - 0.5

    y0 = torch.floor(y_lr_f).long().clamp(0, H_lr - 1)
    x0 = torch.floor(x_lr_f).long().clamp(0, W_lr - 1)
    y1 = (y0 + 1).clamp(0, H_lr - 1)
    x1 = (x0 + 1).clamp(0, W_lr - 1)

    ty = (y_lr_f - y0.float()).clamp(0.0, 1.0)
    tx = (x_lr_f - x0.float()).clamp(0.0, 1.0)
    w_nb = torch.stack(
        [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty],
        dim=-1,
    )
    g_nb = torch.stack(
        [labels_lr[y0, x0], labels_lr[y0, x1], labels_lr[y1, x0], labels_lr[y1, x1]],
        dim=-1,
    )
    same = g_nb == labels_hr.unsqueeze(-1)

    N = H_hr * W_hr
    sf = same.reshape(N, 4)
    wf = w_nb.reshape(N, 4)
    nv = sf.sum(dim=1)
    cf = torch.zeros(N, dtype=torch.uint8, device=dev)
    cf[nv == 4] = 3
    cf[(nv >= 2) & (nv < 4)] = 2
    cf[nv == 1] = 1
    return sf, wf, cf.view(H_hr, W_hr)


@torch.no_grad()
def _ba_slerp_core(q_lr_t, labels_lr, labels_hr, sym_ops_4x4, scale_y, scale_x, same_mask_flat, w_flat, case_map):
    dev = q_lr_t.device
    dtype = q_lr_t.dtype
    _, _, H_lr, W_lr = q_lr_t.shape
    H_hr, W_hr = labels_hr.shape
    N = H_hr * W_hr

    yy = torch.arange(H_hr, device=dev, dtype=torch.float32)
    xx = torch.arange(W_hr, device=dev, dtype=torch.float32)
    y_g, x_g = torch.meshgrid(yy, xx, indexing="ij")
    y_lr_f = (y_g + 0.5) / scale_y - 0.5
    x_lr_f = (x_g + 0.5) / scale_x - 0.5

    y0 = torch.floor(y_lr_f).long().clamp(0, H_lr - 1)
    x0 = torch.floor(x_lr_f).long().clamp(0, W_lr - 1)
    y1 = (y0 + 1).clamp(0, H_lr - 1)
    x1 = (x0 + 1).clamp(0, W_lr - 1)
    u_f = (x_lr_f - x0.float()).clamp(0.0, 1.0).reshape(-1)
    v_f = (y_lr_f - y0.float()).clamp(0.0, 1.0).reshape(-1)

    q_hw = q_lr_t.squeeze(0).permute(1, 2, 0)
    q00_f = q_hw[y0, x0].reshape(N, 4)
    q01_f = q_hw[y0, x1].reshape(N, 4)
    q10_f = q_hw[y1, x0].reshape(N, 4)
    q11_f = q_hw[y1, x1].reshape(N, 4)
    q_nb = torch.stack([q00_f, q01_f, q10_f, q11_f], dim=1)

    cf = case_map.reshape(-1)
    gid_f = labels_hr.reshape(-1)
    y_est_f = y_lr_f.reshape(-1)
    x_est_f = x_lr_f.reshape(-1)
    q_out = torch.zeros((N, 4), dtype=dtype, device=dev)

    idx3 = (cf == 3).nonzero(as_tuple=False).squeeze(1)
    if idx3.numel() > 0:
        q_out[idx3] = bilinear_slerp_sym(
            q00_f[idx3],
            q01_f[idx3],
            q10_f[idx3],
            q11_f[idx3],
            u_f[idx3],
            v_f[idx3],
            sym_ops_4x4,
        )

    idx1 = (cf == 1).nonzero(as_tuple=False).squeeze(1)
    if idx1.numel() > 0:
        pos1 = same_mask_flat[idx1].long().argmax(dim=1)
        q_out[idx1] = q_nb[idx1, pos1]

    idx2 = (cf == 2).nonzero(as_tuple=False).squeeze(1)
    if idx2.numel() > 0:
        for i in range(idx2.numel()):
            valid = same_mask_flat[idx2[i]].nonzero(as_tuple=False).squeeze(1)
            qv = q_nb[idx2[i], valid]
            wv = w_flat[idx2[i], valid]
            wv = wv / (wv.sum() + 1e-12)
            q_ref = qv[0:1]
            total_w = wv[0].clone()
            for k in range(1, valid.numel()):
                alpha = wv[k] / (total_w + wv[k] + 1e-12)
                q_ref, q_nxt = symmetrize_pair(q_ref, qv[k:k + 1], sym_ops_4x4)
                q_ref = _slerp_t(
                    q_ref,
                    q_nxt,
                    torch.tensor([float(alpha)], device=dev, dtype=dtype),
                )
                total_w = total_w + wv[k]
            q_out[idx2[i]] = q_ref[0]

    idx0 = (cf == 0).nonzero(as_tuple=False).squeeze(1)
    if idx0.numel() > 0:
        lab_cpu = labels_lr.cpu()
        q_hw_cpu = q_hw.cpu()
        for j in idx0.tolist():
            gid = int(gid_f[j].item())
            ye = int(round(float(y_est_f[j].item())))
            xe = int(round(float(x_est_f[j].item())))
            best_q = None
            for r in range(1, 4):
                if best_q is not None:
                    break
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        yy_ = min(max(ye + dy, 0), H_lr - 1)
                        xx_ = min(max(xe + dx, 0), W_lr - 1)
                        if int(lab_cpu[yy_, xx_].item()) == gid:
                            best_q = q_hw_cpu[yy_, xx_]
                            break
                    if best_q is not None:
                        break
            if best_q is None:
                best_q = q_hw_cpu[min(max(ye, 0), H_lr - 1), min(max(xe, 0), W_lr - 1)]
            q_out[j] = _qnorm_t(best_q.to(dev, dtype).unsqueeze(0))[0]

    q_hr = _qnorm_t(q_out.view(H_hr, W_hr, 4).permute(2, 0, 1).unsqueeze(0))
    return q_hr


def upsample_ba_sym_slerp(lr_q, out_hw):
    lr_q = _hwc4(lr_q)
    q_lr_cpu = _qnorm_t(torch.from_numpy(lr_q).permute(2, 0, 1).unsqueeze(0).float())
    labels_np, _ = segment_grains_graph(q_lr_cpu, _ba_sym_ops_cpu, thr_deg=3.0)
    labels_lr = torch.from_numpy(labels_np).long()
    if torch.cuda.is_available():
        labels_lr = cleanup_small_grains_cuda(
            labels_lr.to(_ba_device),
            q_lr_cpu.to(_ba_device),
            _ba_sym_ops_cpu.to(_ba_device),
            min_pixels=3,
            max_iter=1,
        ).cpu()
    labels_hr = _build_smooth_hr_labels_aniso(labels_lr, 4, 1, sdf_shift=0.7)
    same_mask_flat, w_flat, case_map = _analyze_boundary_cases_aniso(labels_lr, labels_hr, 4, 1)
    q_hr = _ba_slerp_core(
        q_lr_cpu.to(_ba_device),
        labels_lr.to(_ba_device),
        labels_hr.to(_ba_device),
        _ba_sym_ops_cpu.to(_ba_device),
        4,
        1,
        same_mask_flat.to(_ba_device),
        w_flat.to(_ba_device),
        case_map.to(_ba_device),
    )
    return q_hr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


METHODS = OrderedDict([
    ("Model SR", upsample_model_sr),
    ("Boundary-aware SR", upsample_boundary_aware_model_sr),
    ("SDF 1sided SR", upsample_sdf_one_sided_sr),
    ("Bicubic", upsample_bicubic),
    ("Bicubic (F.interpolate)", upsample_bicubic_finterp),
    ("ISO Bicubic", upsample_iso_bicubic),
    ("NN", upsample_nn),
    ("ISO NN", upsample_iso_nn),
    ("SLERP", upsample_slerp),
    ("Symm-SLERP", upsample_symm_slerp),
    ("BA Sym-SLERP", upsample_ba_sym_slerp),
    ("QRBSA", upsample_qrbsa_sr),
])

print("Methods:", list(METHODS.keys()))

BASE_MODEL_METHODS = {"Model SR", "ISO Bicubic", "ISO NN"}
BOUNDARY_AWARE_MODEL_METHODS = {"Boundary-aware SR"}
SDF_ONE_SIDED_MODEL_METHODS = {"SDF 1sided SR"}
QRBSA_MODEL_METHODS = {"QRBSA"}


def _move_model(model_obj, target_device):
    model_obj.to(target_device)
    if hasattr(model_obj, 'device'):
        model_obj.device = torch.device(target_device)
    model_obj.eval()


def _prepare_models_for_method(method_name):
    if not torch.cuda.is_available():
        return
    if method_name in BASE_MODEL_METHODS:
        _ensure_base_model_loaded()
        _move_model(model, device)
        _unload_boundary_aware_model()
        _unload_sdf_one_sided_model()
        _unload_qrbsa_model()
    elif method_name in BOUNDARY_AWARE_MODEL_METHODS:
        _ensure_boundary_aware_model_loaded()
        _move_model(model_boundary_aware, device)
        _unload_base_model()
        _unload_sdf_one_sided_model()
        _unload_qrbsa_model()
    elif method_name in SDF_ONE_SIDED_MODEL_METHODS:
        _ensure_sdf_one_sided_model_loaded()
        _move_model(model_sdf_one_sided, device)
        _unload_base_model()
        _unload_boundary_aware_model()
        _unload_qrbsa_model()
    elif method_name in QRBSA_MODEL_METHODS:
        _ensure_qrbsa_model_loaded()
        _move_model(model_qrbsa, device)
        _unload_base_model()
        _unload_boundary_aware_model()
        _unload_sdf_one_sided_model()
    else:
        _unload_base_model()
        _unload_boundary_aware_model()
        _unload_sdf_one_sided_model()
        _unload_qrbsa_model()
        gc.collect()
    torch.cuda.empty_cache()

# %% Section 5: Full-test evaluation harness
def _summarize_global_maps(mis_concat, grod_sr_concat, grod_hr_concat, kam_sr_concat, kam_hr_concat, max_mis_deg):
    zero_ref = np.zeros(mis_concat.shape, dtype=np.float32)

    mis_stats = {
        "mean": float(np.nanmean(mis_concat)),
        "median": float(np.nanmedian(mis_concat)),
        "p90": float(np.nanpercentile(mis_concat, 90)),
        "psnr": psnr_from_map(zero_ref, mis_concat, max_val=max_mis_deg),
        "ssim": ssim_from_map(zero_ref, mis_concat, win_size=7, data_range=max_mis_deg),
    }

    grod_stats = _summarize_scalar_map_pair(
        grod_sr_concat,
        grod_hr_concat,
        percentile_key="p90",
        percentile=90,
    )
    kam_stats = _summarize_scalar_map_pair(
        kam_sr_concat,
        kam_hr_concat,
        percentile_key="p90",
        percentile=90,
    )

    return {
        "mis": mis_stats,
        "grod": grod_stats,
        "kam": kam_stats,
        "concat_shape": tuple(mis_concat.shape),
    }


def evaluate_method_over_dataset(
    method_name,
    upsample_fn,
    pairs,
    *,
    representative_index=0,
    patch_rows=PATCH_ROWS,
    patch_cols=PATCH_COLS,
    sym_ops=sym,
):
    max_mis_deg = max_misorientation_from_sym(sym_ops)
    representative = None
    cache = _prepare_resume_cache(
        method_name,
        pairs,
        representative_index,
        patch_rows,
        patch_cols,
        sym_ops,
    )
    cache_state = cache["state"]
    if (
        cache_state.get("status") == "completed"
        and cache["final_result_path"].exists()
        and cache["representative_path"].exists()
    ):
        representative = _load_representative_cache(cache["representative_path"])
        global_stats = _read_json(cache["final_result_path"])["global"]
        print(f"[{method_name}] loaded completed evaluation from resume cache: {cache['cache_dir']}")
        return {
            "representative": representative,
            "global": global_stats,
        }

    _prepare_models_for_method(method_name)

    array_paths = cache["array_paths"]
    arrays_exist = all(path.exists() for path in array_paths.values())
    array_mode = "r+" if arrays_exist else "w+"
    mis_concat = np.memmap(array_paths["mis"], dtype=np.float32, mode=array_mode, shape=CONCAT_SHAPE)
    grod_sr_concat = np.memmap(array_paths["grod_sr"], dtype=np.float32, mode=array_mode, shape=CONCAT_SHAPE)
    grod_hr_concat = np.memmap(array_paths["grod_hr"], dtype=np.float32, mode=array_mode, shape=CONCAT_SHAPE)
    kam_sr_concat = np.memmap(array_paths["kam_sr"], dtype=np.float32, mode=array_mode, shape=CONCAT_SHAPE)
    kam_hr_concat = np.memmap(array_paths["kam_hr"], dtype=np.float32, mode=array_mode, shape=CONCAT_SHAPE)

    if array_mode == "w+":
        for arr in (mis_concat, grod_sr_concat, grod_hr_concat, kam_sr_concat, kam_hr_concat):
            arr[:] = np.nan
            arr.flush()

    if cache["representative_path"].exists():
        representative = _load_representative_cache(cache["representative_path"])

    start_idx = int(cache_state.get("completed_samples", 0))
    row_start = int(cache_state.get("row_start", 0))
    if 0 < start_idx < len(pairs):
        print(f"[{method_name}] resuming from sample {start_idx + 1}/{len(pairs)} using {cache['cache_dir']}")

    try:
        for idx in tqdm(range(start_idx, len(pairs)), desc=method_name, leave=False):
            pair = pairs[idx]
            lr = np.load(pair["lr_file"]).astype(np.float32)
            hr = np.load(pair["hr_file"]).astype(np.float32)

            sr_full = upsample_fn(lr, hr.shape[:2])
            mis_full = misorientation_map(sr_full, hr, sym_quats=sym_ops, degrees=True).astype(np.float32)
            grod_sr_full, _ = compute_grod(sr_full, sym_ops=sym_ops)
            grod_hr_full, _ = compute_grod(hr, sym_ops=sym_ops)
            kam_sr_full, _ = compute_kam(sr_full, radius=1, sym_ops=sym_ops)
            kam_hr_full, _ = compute_kam(hr, radius=1, sym_ops=sym_ops)

            h, w = hr.shape[:2]
            row_end = row_start + h
            mis_concat[row_start:row_end, :w] = mis_full
            grod_sr_concat[row_start:row_end, :w] = grod_sr_full.astype(np.float32)
            grod_hr_concat[row_start:row_end, :w] = grod_hr_full.astype(np.float32)
            kam_sr_concat[row_start:row_end, :w] = kam_sr_full.astype(np.float32)
            kam_hr_concat[row_start:row_end, :w] = kam_hr_full.astype(np.float32)
            for arr in (mis_concat, grod_sr_concat, grod_hr_concat, kam_sr_concat, kam_hr_concat):
                arr.flush()

            row_start = row_end

            if idx == representative_index:
                representative = {
                    "sample_id": pair["sample_id"],
                    "lr": lr,
                    "hr": hr,
                    "hr_patch": hr[patch_rows, patch_cols],
                    "sr_full": sr_full,
                    "sr_patch": sr_full[patch_rows, patch_cols],
                    "mis_patch": mis_full[patch_rows, patch_cols],
                    "grod": grod_sr_full[patch_rows, patch_cols].astype(np.float32),
                    "kam": kam_sr_full[patch_rows, patch_cols].astype(np.float32),
                }
                _save_representative_cache(cache["representative_path"], representative)

            _write_method_state(
                cache["state_path"],
                signature=cache["signature"],
                status="running",
                completed_samples=idx + 1,
                row_start=row_start,
                last_sample_id=pair["sample_id"],
            )

            del lr, hr, sr_full, mis_full, grod_sr_full, grod_hr_full, kam_sr_full, kam_hr_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if representative is None and 0 <= representative_index < len(pairs):
            pair = pairs[representative_index]
            lr = np.load(pair["lr_file"]).astype(np.float32)
            hr = np.load(pair["hr_file"]).astype(np.float32)
            sr_full = upsample_fn(lr, hr.shape[:2])
            mis_full = misorientation_map(sr_full, hr, sym_quats=sym_ops, degrees=True).astype(np.float32)
            grod_sr_full, _ = compute_grod(sr_full, sym_ops=sym_ops)
            kam_sr_full, _ = compute_kam(sr_full, radius=1, sym_ops=sym_ops)
            representative = {
                "sample_id": pair["sample_id"],
                "lr": lr,
                "hr": hr,
                "hr_patch": hr[patch_rows, patch_cols],
                "sr_full": sr_full,
                "sr_patch": sr_full[patch_rows, patch_cols],
                "mis_patch": mis_full[patch_rows, patch_cols],
                "grod": grod_sr_full[patch_rows, patch_cols].astype(np.float32),
                "kam": kam_sr_full[patch_rows, patch_cols].astype(np.float32),
            }
            _save_representative_cache(cache["representative_path"], representative)
            del lr, hr, sr_full, mis_full, grod_sr_full, kam_sr_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_stats = _summarize_global_maps(
            mis_concat=mis_concat,
            grod_sr_concat=grod_sr_concat,
            grod_hr_concat=grod_hr_concat,
            kam_sr_concat=kam_sr_concat,
            kam_hr_concat=kam_hr_concat,
            max_mis_deg=max_mis_deg,
        )
        _write_json(
            cache["final_result_path"],
            {
                "method_name": method_name,
                "global": global_stats,
                "representative_sample_id": None if representative is None else representative["sample_id"],
            },
            verbose=False,
        )
        _write_method_state(
            cache["state_path"],
            signature=cache["signature"],
            status="completed",
            completed_samples=len(pairs),
            row_start=CONCAT_SHAPE[0],
            last_sample_id=None if not pairs else pairs[-1]["sample_id"],
        )
    finally:
        del mis_concat, grod_sr_concat, grod_hr_concat, kam_sr_concat, kam_hr_concat

    print(
        f"[{method_name}] "
        f"global mis_mean={global_stats['mis']['mean']:.3f}°, "
        f"PSNR={global_stats['mis']['psnr']:.2f} dB, "
        f"SSIM={global_stats['mis']['ssim']:.4f}, "
        f"mean|GROD-HR|={global_stats['grod']['mean_abs_delta']:.3f}°, "
        f"mean|KAM-HR|={global_stats['kam']['mean_abs_delta']:.3f}°"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _unload_base_model()
    _unload_boundary_aware_model()
    _unload_sdf_one_sided_model()
    _unload_qrbsa_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "representative": representative,
        "global": global_stats,
    }


def compare_methods_over_dataset(
    method_a_name,
    upsample_fn_a,
    method_b_name,
    upsample_fn_b,
    pairs,
    *,
    representative_index=0,
    patch_rows=PATCH_ROWS,
    patch_cols=PATCH_COLS,
    sym_ops=sym,
):
    representative = None
    safe_pair_name = (
        f"{method_a_name}_vs_{method_b_name}".lower().replace(" ", "_").replace("/", "_")
    )
    with tempfile.TemporaryDirectory(prefix=f"dataset_cmp_{safe_pair_name}_", dir=ARTIFACT_ROOT) as tmpdir:
        tmpdir = Path(tmpdir)
        mis_concat = np.memmap(tmpdir / "mis.dat", dtype=np.float32, mode="w+", shape=CONCAT_SHAPE)
        row_start = 0
        abs_component_sum = 0.0
        component_count = 0

        for idx, pair in enumerate(tqdm(pairs, desc=f"{method_a_name} vs {method_b_name}", leave=False)):
            lr = np.load(pair["lr_file"]).astype(np.float32)
            hr = np.load(pair["hr_file"]).astype(np.float32)

            sr_a = upsample_fn_a(lr, hr.shape[:2])
            sr_b = upsample_fn_b(lr, hr.shape[:2])
            mis_full = misorientation_map(sr_a, sr_b, sym_quats=sym_ops, degrees=True).astype(np.float32)

            h, w = mis_full.shape
            row_end = row_start + h
            mis_concat[row_start:row_end, :w] = mis_full
            row_start = row_end

            abs_component_sum += float(np.abs(sr_a - sr_b).sum())
            component_count += int(sr_a.size)

            if idx == representative_index:
                rep_patch_a = sr_a[patch_rows, patch_cols]
                rep_patch_b = sr_b[patch_rows, patch_cols]
                rep_patch_mis = mis_full[patch_rows, patch_cols]
                representative = {
                    "sample_id": pair["sample_id"],
                    "summary": {
                        "mean": float(np.nanmean(rep_patch_mis)),
                        "median": float(np.nanmedian(rep_patch_mis)),
                        "p90": float(np.nanpercentile(rep_patch_mis, 90)),
                        "max": float(np.nanmax(rep_patch_mis)),
                        "mean_abs_component_diff": float(np.mean(np.abs(rep_patch_a - rep_patch_b))),
                    },
                }

            del lr, hr, sr_a, sr_b, mis_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mis_concat.flush()
        dataset_summary = {
            "mean": float(np.nanmean(mis_concat)),
            "median": float(np.nanmedian(mis_concat)),
            "p90": float(np.nanpercentile(mis_concat, 90)),
            "max": float(np.nanmax(mis_concat)),
            "mean_abs_component_diff": float(abs_component_sum / max(component_count, 1)),
            "concat_shape": tuple(mis_concat.shape),
        }

    return {
        "method_a": method_a_name,
        "method_b": method_b_name,
        "dataset": dataset_summary,
        "representative": representative,
    }


# %% Section 6: Run full test-set comparison
all_results = OrderedDict()
for method_name, method_fn in METHODS.items():
    all_results[method_name] = evaluate_method_over_dataset(
        method_name,
        method_fn,
        PAIRS,
        representative_index=REPRESENTATIVE_INDEX,
        patch_rows=PATCH_ROWS,
        patch_cols=PATCH_COLS,
        sym_ops=sym,
    )
    _write_json(
        METRIC_DIR / "global_metrics_progress.json",
        _progress_metrics_payload(all_results),
        verbose=False,
    )

representative_sample_id = REP_PAIR["sample_id"]
print("Finished full test-set evaluation for:", len(all_results), "methods")
print("Representative sample used for plots:", representative_sample_id)

bicubic_backend_comparison = None
if "Bicubic" in METHODS and "Bicubic (F.interpolate)" in METHODS:
    bicubic_backend_comparison = compare_methods_over_dataset(
        "Bicubic",
        METHODS["Bicubic"],
        "Bicubic (F.interpolate)",
        METHODS["Bicubic (F.interpolate)"],
        PAIRS,
        representative_index=REPRESENTATIVE_INDEX,
        patch_rows=PATCH_ROWS,
        patch_cols=PATCH_COLS,
        sym_ops=sym,
    )
    _write_json(METRIC_DIR / "bicubic_backend_comparison.json", bicubic_backend_comparison)
    bicubic_ds = bicubic_backend_comparison["dataset"]
    print(
        "[Bicubic backend comparison] "
        f"mean={bicubic_ds['mean']:.4f}°, "
        f"p90={bicubic_ds['p90']:.4f}°, "
        f"max={bicubic_ds['max']:.4f}°, "
        f"mean_abs_component_diff={bicubic_ds['mean_abs_component_diff']:.6f}"
    )

run_metadata = {
    "timestamp": RUN_STAMP,
    "output_root": OUTPUT_ROOT,
    "dataset_dir": DATASET_DIR,
    "split": SPLIT,
    "n_samples": len(PAIRS),
    "representative_sample_id": representative_sample_id,
    "concat_shape_for_dataset_metrics": CONCAT_SHAPE,
    "device": str(device),
    "device_info": device_info,
    "misorientation_metric_max_deg": max_misorientation_from_sym(sym),
    "resume_cache_root": RESUME_CACHE_ROOT,
    "config_batch_size_override": CONFIG_BATCH_SIZE,
    "show_plots": SHOW_PLOTS,
    "display_plots": DISPLAY_PLOTS,
    "save_plots": SAVE_PLOTS,
    "methods": list(METHODS.keys()),
}
global_metrics_payload = _progress_metrics_payload(all_results)
_write_json(METRIC_DIR / "run_metadata.json", run_metadata)
_write_json(METRIC_DIR / "global_metrics.json", global_metrics_payload)

# %% Section 7: Full-map IPF overview (representative sample)
if SHOW_PLOTS:
    ref_dirs = ("X", "Y", "Z")
    overview_key = "Model SR" if "Model SR" in all_results else list(METHODS.keys())[0]
    rep_sr_full = all_results[overview_key]["representative"]["sr_full"]

    lr_ipf = dict(zip(ref_dirs, render_ipf_rgb(rep_lr, sym)))
    hr_ipf = dict(zip(ref_dirs, render_ipf_rgb(rep_hr, sym)))
    sr_ipf = dict(zip(ref_dirs, render_ipf_rgb(rep_sr_full, sym)))

    overview_cols = ["LR", "HR", overview_key]
    overview_maps = [lr_ipf, hr_ipf, sr_ipf]
    overview_row_labels = ["Full IPF-X", "Full IPF-Y", "Full IPF-Z"]

    fig = plt.figure(figsize=(12.6, 9.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, width_ratios=[0.34, 1, 1, 1])
    fig.suptitle(
        f"{representative_sample_id} — Full-map IPF Overview (SR = {overview_key})",
        fontsize=15,
    )

    for r, rd in enumerate(ref_dirs):
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        ax_label.text(0.98, 0.5, overview_row_labels[r], ha="right", va="center", fontsize=10)

        for c, ipf_map in enumerate(overview_maps, start=1):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(ipf_map[rd])
            ax.axis("off")
            if r == 0:
                ax.set_title(overview_cols[c - 1], fontsize=11, pad=8)

    if SAVE_PLOTS:
        _finalize_figure(fig, FIGURE_DIR / "01_full_map_ipf_overview.png")
    else:
        plt.close(fig)

# %% Section 8: Combined representative patch dashboard
if SHOW_PLOTS:
    from matplotlib.colors import Normalize

    method_names = list(METHODS.keys())
    ref_dirs = ("X", "Y", "Z")
    panel_names = ["HR Patch"] + method_names
    n_methods = len(method_names)
    n_data_cols = len(panel_names)

    patch_ipf = {"HR Patch": dict(zip(ref_dirs, render_ipf_rgb(rep_hr_patch, sym)))}
    for name in method_names:
        patch_ipf[name] = dict(zip(ref_dirs, render_ipf_rgb(all_results[name]["representative"]["sr_patch"], sym)))

    vmax_grod = max(
        [np.nanmax(rep_grod_hr_patch)] + [np.nanmax(all_results[n]["representative"]["grod"]) for n in method_names]
    )
    vmax_kam = max(
        [np.nanmax(rep_kam_hr_patch)] + [np.nanmax(all_results[n]["representative"]["kam"]) for n in method_names]
    )

    row_labels = [
        "Patch IPF-X",
        "Patch IPF-Y",
        "Patch IPF-Z",
        "Patch GROD",
        "Patch KAM",
        "Patch Misorientation\nHistogram",
    ]

    fig = plt.figure(figsize=(2.2 * (n_data_cols + 1.45), 15.3), constrained_layout=True)
    gs = fig.add_gridspec(
        6,
        n_data_cols + 2,
        width_ratios=[0.48] + [1] * n_data_cols + [0.08],
        height_ratios=[1, 1, 1, 1, 1, 1.24],
    )
    fig.suptitle(f"{representative_sample_id} — Patch Comparison Dashboard", fontsize=17)

    for r, label in enumerate(row_labels):
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        ax_label.text(0.98, 0.5, label, ha="right", va="center", fontsize=9)

    axes = []
    for r in range(5):
        row_axes = []
        for c, panel_name in enumerate(panel_names, start=1):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if r == 0:
                ax.set_title(panel_name, fontsize=8, pad=4)
            row_axes.append(ax)
        axes.append(row_axes)

    for ridx, rd in enumerate(ref_dirs):
        axes[ridx][0].imshow(patch_ipf["HR Patch"][rd])
        for c, name in enumerate(method_names, start=1):
            axes[ridx][c].imshow(patch_ipf[name][rd])

    norm_grod = Normalize(vmin=0, vmax=vmax_grod)
    for c in range(n_data_cols):
        data = rep_grod_hr_patch if c == 0 else all_results[method_names[c - 1]]["representative"]["grod"]
        im_grod = axes[3][c].imshow(data, cmap="inferno", norm=norm_grod)
    grod_cax = fig.add_subplot(gs[3, -1])
    fig.colorbar(im_grod, cax=grod_cax, label="GROD (°)")

    norm_kam = Normalize(vmin=0, vmax=vmax_kam)
    for c in range(n_data_cols):
        data = rep_kam_hr_patch if c == 0 else all_results[method_names[c - 1]]["representative"]["kam"]
        im_kam = axes[4][c].imshow(data, cmap="coolwarm", norm=norm_kam)
    kam_cax = fig.add_subplot(gs[4, -1])
    fig.colorbar(im_kam, cax=kam_cax, label="KAM (°), blue → red")

    ax_hist_ref = fig.add_subplot(gs[5, 1])
    ax_hist_ref.axis("off")
    ax_hist_ref.text(0.5, 0.57, "HR patch\nreference", ha="center", va="center", fontsize=9, color="0.35")
    ax_hist_ref.text(
        0.5,
        0.29,
        "No histogram here because\nmisorientation is defined vs HR.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.45",
    )

    hist_gs = gs[5, 2 : n_data_cols + 1].subgridspec(1, n_methods, wspace=0.14)
    hist_colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    for idx, name in enumerate(method_names):
        ax = fig.add_subplot(hist_gs[0, idx])
        m = all_results[name]["representative"]["mis_patch"].ravel()
        m = m[~np.isnan(m)]
        m_max = float(np.nanmax(m)) if m.size else 1.0
        x_hi = max(1.0, m_max * 1.02)
        bins = np.linspace(0.0, x_hi, 41)
        ax.hist(
            m,
            bins=bins,
            density=True,
            color=hist_colors[idx % len(hist_colors)],
            alpha=0.84,
        )
        ax.axvline(np.nanmean(m), color="black", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, x_hi)
        ax.tick_params(labelsize=7)
        ax.set_title(name, fontsize=7, pad=2)
        ax.set_xlabel("deg", fontsize=7)
        ax.text(
            0.98,
            0.94,
            f"max={m_max:.2f}°",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="0.35",
        )
        if idx == 0:
            ax.set_ylabel("density", fontsize=7)
        else:
            ax.set_yticklabels([])

    if SAVE_PLOTS:
        _finalize_figure(fig, FIGURE_DIR / "02_patch_comparison_dashboard.png")
    else:
        plt.close(fig)

# %% Section 9: Full test-split scalar metrics summary
method_names = list(METHODS.keys())

rows = []
for name in method_names:
    g = all_results[name]["global"]
    rows.append(
        {
            "Method": name,
            "Mis mean": g["mis"]["mean"],
            "Mis median": g["mis"]["median"],
            "Mis p90": g["mis"]["p90"],
            "PSNR mis": g["mis"]["psnr"],
            "SSIM mis": g["mis"]["ssim"],
            "GROD mean": g["grod"]["mean"],
            "GROD std": g["grod"]["std"],
            "mean|GROD - HR|": g["grod"]["mean_abs_delta"],
            "GROD p90": g["grod"]["p90"],
            "KAM mean": g["kam"]["mean"],
            "KAM std": g["kam"]["std"],
            "mean|KAM - HR|": g["kam"]["mean_abs_delta"],
            "KAM p90": g["kam"]["p90"],
        }
    )

df_summary = pd.DataFrame(rows).set_index("Method").T
metric_order = [
    "Mis mean",
    "Mis median",
    "Mis p90",
    "PSNR mis",
    "SSIM mis",
    "GROD mean",
    "GROD std",
    "mean|GROD - HR|",
    "GROD p90",
    "KAM mean",
    "KAM std",
    "mean|KAM - HR|",
    "KAM p90",
]
df_summary = df_summary.loc[metric_order]

first_method = method_names[0]
reference_stats = pd.DataFrame(
    {
        "Whole test split HR": {
            "GROD mean": all_results[first_method]["global"]["grod"]["hr_mean"],
            "GROD std": all_results[first_method]["global"]["grod"]["hr_std"],
            "GROD p90": all_results[first_method]["global"]["grod"]["hr_p90"],
            "KAM mean": all_results[first_method]["global"]["kam"]["hr_mean"],
            "KAM std": all_results[first_method]["global"]["kam"]["hr_std"],
            "KAM p90": all_results[first_method]["global"]["kam"]["hr_p90"],
        }
    }
)
metric_targets = reference_stats["Whole test split HR"].to_dict()

metric_pref = {
    "Mis mean": "min",
    "Mis median": "min",
    "Mis p90": "min",
    "PSNR mis": "max",
    "SSIM mis": "max",
    "GROD mean": "ref",
    "GROD std": "ref",
    "mean|GROD - HR|": "min",
    "GROD p90": "ref",
    "KAM mean": "ref",
    "KAM std": "ref",
    "mean|KAM - HR|": "min",
    "KAM p90": "ref",
}

quick_rows = []
for metric in df_summary.index:
    vals = pd.to_numeric(df_summary.loc[metric], errors="coerce")
    direction = metric_pref[metric]
    if direction == "max":
        best_method = vals.idxmax()
        best_value = float(np.nanmax(vals.to_numpy(dtype=float)))
        better = "higher"
    elif direction == "ref":
        target = float(metric_targets[metric])
        best_method = (vals - target).abs().idxmin()
        best_value = float(vals.loc[best_method])
        better = "closest to HR"
    else:
        best_method = vals.idxmin()
        best_value = float(np.nanmin(vals.to_numpy(dtype=float)))
        better = "lower"
    quick_rows.append(
        {
            "Metric": metric,
            "Better": better,
            "Best method": best_method,
            "Best value": best_value,
        }
    )
quick_read = pd.DataFrame(quick_rows).set_index("Metric")

print("Full test-split scalar summary:")
print("- lower is better for misorientation rows and for mean|GROD - HR| / mean|KAM - HR|")
print("- GROD/KAM mean/std/percentile rows are ranked by closeness to the HR reference table")
print("- PSNR/SSIM are computed from the whole test split, not averaged per sample")
print("- GROD/KAM summary rows are computed from concatenated per-sample scalar maps")
print("- mean|GROD - HR| and mean|KAM - HR| are direct whole-split scalar-map MAEs")
print("- representative sample for plots:", representative_sample_id)
print("- test samples used:", len(PAIRS))
print("- concatenated misorientation map shape:", all_results[first_method]["global"]["concat_shape"])
if not SKIMAGE_AVAILABLE:
    print("- SSIM unavailable because scikit-image is not installed in this kernel")


def _highlight_by_direction(row):
    vals = row.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return [""] * len(vals)
    direction = metric_pref[row.name]
    if direction == "max":
        score = vals.copy()
        best = np.nanmax(score)
        worst = np.nanmin(score)
    elif direction == "ref":
        target = float(metric_targets[row.name])
        score = np.abs(vals - target)
        best = np.nanmin(score)
        worst = np.nanmax(score)
    else:
        score = vals.copy()
        best = np.nanmin(score)
        worst = np.nanmax(score)
    styles = []
    finite_count = np.count_nonzero(finite)
    for v, s in zip(vals, score):
        if not np.isfinite(v):
            styles.append("color: #777777;")
        elif np.isclose(s, best, equal_nan=False):
            styles.append("background-color: #e8f5e9; font-weight: 700; border: 2px solid #2e7d32;")
        elif np.isclose(s, worst, equal_nan=False) and finite_count > 1:
            styles.append("background-color: #fdecea;")
        else:
            styles.append("")
    return styles


reference_style = (
    reference_stats.style.format("{:.3f}").set_caption("Whole test-split HR reference scalar stats").set_table_styles(
        [
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "12px"), ("font-weight", "600")]},
            {"selector": "th", "props": [("text-align", "center"), ("padding", "6px 10px"), ("background-color", "#f4f6f8")]},
            {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 10px")]},
        ]
    )
)
quick_read_style = (
    quick_read.style.format({"Best value": "{:.3f}"}).set_caption("Quick read: best-performing method by metric").set_table_styles(
        [
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "12px"), ("font-weight", "600")]},
            {"selector": "th", "props": [("text-align", "center"), ("padding", "6px 10px"), ("background-color", "#f4f6f8")]},
            {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 10px")]},
        ]
    )
)
styled_summary = (
    df_summary.style.format("{:.3f}")
    .apply(_highlight_by_direction, axis=1)
    .set_caption("Whole test-split scalar metrics summary")
    .set_table_styles(
        [
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "13px"), ("font-weight", "600")]},
            {"selector": "th", "props": [("text-align", "center"), ("padding", "6px 10px"), ("background-color", "#f4f6f8")]},
            {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 10px")]},
        ]
    )
)

_save_dataframe_bundle(
    reference_stats,
    TABLE_DIR / "reference_stats.csv",
    TABLE_DIR / "reference_stats.html",
    styler=reference_style,
    float_format="%.6f",
)
_save_dataframe_bundle(
    quick_read,
    TABLE_DIR / "quick_read.csv",
    TABLE_DIR / "quick_read.html",
    styler=quick_read_style,
    float_format="%.6f",
)
_save_dataframe_bundle(
    df_summary,
    TABLE_DIR / "summary_metrics.csv",
    TABLE_DIR / "summary_metrics.html",
    styler=styled_summary,
    float_format="%.6f",
)
_write_json(METRIC_DIR / "summary_metrics.json", df_summary.to_dict())
_write_json(METRIC_DIR / "quick_read.json", quick_read.reset_index().to_dict(orient="records"))
_write_json(METRIC_DIR / "reference_stats.json", reference_stats.to_dict())

print("\nRaw summary dataframe:")
print(df_summary.round(4))
print("\nSaved outputs under:", OUTPUT_ROOT)
