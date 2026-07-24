import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from autoencoder_hcp import HCPAutoEncoder
from utils.quat_ops import to_spatial_quat
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_input_output_side_by_side


# -----------------------------------------------------------------------------
# Simple config (edit if needed)
# -----------------------------------------------------------------------------
DATASET_ROOT = Path(
    "/data/warren/materials/materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4"
)
LOOKUP_PATH = PROJECT_ROOT / "symmetry_groups" / "local_iso_lookup_D6_z_axis_res1_irreps.npy"
OUT_DIR = PROJECT_ROOT / "out" / "local_iso_hcp_ipf" / "test_ds_first10"

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))
# Full HR by default. Optional env override: CROP_HW=H,W
_crop_hw_env = os.environ.get("CROP_HW", "none").strip()
if _crop_hw_env.lower() in {"", "none", "full"}:
    CROP_HW = None
else:
    h_str, w_str = _crop_hw_env.split(",")
    CROP_HW = (int(h_str), int(w_str))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECODE_CHUNK = int(os.environ.get("DECODE_CHUNK", "2048" if DEVICE == "cpu" else "8192"))


def _misorientation_deg(q_ref: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
    q_ref = q_ref / q_ref.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = (q_ref * q_pred).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(w) * 180.0 / torch.pi


@torch.no_grad()
def run_one(
    model: HCPAutoEncoder,
    hr_fp: Path,
    sym_class,
    out_dir: Path,
) -> dict[str, float | str | int]:
    hr_arr = np.load(hr_fp, mmap_mode="r")
    q_hw4 = to_spatial_quat(np.array(hr_arr, copy=True))

    if CROP_HW is not None:
        h, w = CROP_HW
        q_hw4 = q_hw4[:h, :w, :]

    h, w, _ = q_hw4.shape
    q_in = torch.from_numpy(q_hw4.reshape(-1, 4).copy()).to(
        device=model.device,
        dtype=torch.float32,
    )

    q_out_parts = []
    for start in range(0, q_in.shape[0], DECODE_CHUNK):
        end = min(start + DECODE_CHUNK, q_in.shape[0])
        f4, f6 = model.encode(q_in[start:end])
        q_out_parts.append(model.decode(f4, f6))
    q_out = torch.cat(q_out_parts, dim=0)

    q_in_fz = model.reduce_to_fz(q_in)
    mis_deg = _misorientation_deg(q_in_fz, q_out)

    q_in_fz_np = q_in_fz.cpu().numpy().reshape(h, w, 4).astype(np.float32, copy=False)
    q_out_np = q_out.cpu().numpy().reshape(h, w, 4).astype(np.float32, copy=False)

    out_png = out_dir / f"{hr_fp.stem}_input_output_ipf_all.png"
    render_input_output_side_by_side(
        input_q_arr=q_in_fz_np,
        output_q_arr=q_out_np,
        sym_class=sym_class,
        out_png=str(out_png),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
    )

    return {
        "file": str(hr_fp),
        "h": int(h),
        "w": int(w),
        "num_quats": int(q_in.shape[0]),
        "mis_deg_mean": float(mis_deg.mean().item()),
        "mis_deg_max": float(mis_deg.max().item()),
        "out_png": str(out_png),
    }


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    hr_dir = DATASET_ROOT / "Test" / "HR_Data"
    test_ds = sorted(hr_dir.glob("*.npy"))[:NUM_SAMPLES]
    if len(test_ds) == 0:
        raise RuntimeError(f"No files found in {hr_dir}")

    model = HCPAutoEncoder(
        device=DEVICE,
        decoder_backend="local_iso_lookup",
        decoder_config={
            "encoder_backend": "local_iso",
            "local_iso_d6_convention": "z_axis",
            "decoder_lookup_npy_path": str(LOOKUP_PATH),
            "decoder_lookup_chunk_size": 16384 if DEVICE == "cuda" else 8192,
        },
    ).eval()

    sym_class = resolve_symmetry("D6h")

    rows = []
    t0 = time.perf_counter()
    print(f"device={DEVICE} num_samples={len(test_ds)} crop={CROP_HW} decode_chunk={DECODE_CHUNK}")
    for i, hr_fp in enumerate(test_ds, start=1):
        s0 = time.perf_counter()
        row = run_one(model, hr_fp, sym_class, out_dir)
        dt = time.perf_counter() - s0
        rows.append(row)
        print(
            f"[{i:02d}/{len(test_ds):02d}] "
            f"mis_mean={row['mis_deg_mean']:.4f}° "
            f"mis_max={row['mis_deg_max']:.4f}° "
            f"time={dt:.2f}s\n"
            f"  file={Path(row['file']).name}\n"
            f"  png ={Path(row['out_png']).name}"
        )

    csv_path = out_dir / "summary_first10_test_ds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "h",
                "w",
                "num_quats",
                "mis_deg_mean",
                "mis_deg_max",
                "out_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = time.perf_counter() - t0
    mean_of_means = float(np.mean([r["mis_deg_mean"] for r in rows]))
    max_of_max = float(np.max([r["mis_deg_max"] for r in rows]))

    print("\n=== Done ===")
    print(f"samples       : {len(rows)}")
    print(f"mean(mis_mean): {mean_of_means:.4f}°")
    print(f"max(mis_max)  : {max_of_max:.4f}°")
    print(f"summary_csv   : {csv_path}")
    print(f"out_dir       : {out_dir}")
    print(f"total_time_s  : {total:.2f}")


if __name__ == "__main__":
    main()
