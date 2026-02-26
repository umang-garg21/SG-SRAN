#!/usr/bin/env python3
"""
Extract pixels with reconstruction error > 0.05 rad, write CSV,
and produce an overlay + thumbnails of bad pixels.

Save location:
experiments/IN718/debug_x4/visualizations/final/bad_pixels.csv
experiments/IN718/debug_x4/visualizations/final/bad_pixels_overlay.png
experiments/IN718/debug_x4/visualizations/final/bad_<idx>_<r>_<c>.png
"""
import sys
import csv
from pathlib import Path
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from orix.quaternion import Orientation, symmetry as orix_sym
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    ORIX_AVAILABLE = True
except Exception:
    ORIX_AVAILABLE = False

# Ensure repository root is on sys.path so local packages import reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from models.autoencoder import FCCAutoEncoder

# user parameters
EXP_DIR = Path("experiments/IN718/debug_x4")
OUT_DIR = EXP_DIR / "visualizations" / "final"
THRESH_RAD = 0.05
PATCH_HALF = 6  # patch radius for thumbnails (13x13)
MAX_THUMBS = 2  # limit thumbnails per sample
PROCESS_ALL = False  # set to True to process whole validation set
INPUT_ORDER = "wxyz"  # 'wxyz' (scalar-first) or 'xyzw' (scalar-last). Set to 'wxyz' per dataset.

OUT_DIR.mkdir(parents=True, exist_ok=True)



def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.stack([q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3], q_wxyz[..., 0]], axis=-1)


def ipf_rgb_orix(q_wxyz_hw4: np.ndarray, laue_sym, ref_dir="Z") -> np.ndarray:
    H, W, _ = q_wxyz_hw4.shape
    ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
    ckey = IPFColorKeyTSL(laue_sym)
    ckey.direction = Vector3d((0, 0, 1)) if ref_dir.upper() == "Z" else Vector3d((1, 0, 0))
    return ckey.orientation2color(ori)


def main():
    cfg = load_and_prepare_config(EXP_DIR / "config.json", EXP_DIR / "logs/run_config.json")
    # build validation loader (process whole split if PROCESS_ALL True)
    loaders = {
        "Test": build_dataloader(
            dataset_root=cfg.dataset_root,
            split="Test",
            batch_size=1,
            num_workers=0,
            preload=False,
            preload_torch=False,
            pin_memory=False,
            take_first=None if PROCESS_ALL else 1,
            seed=getattr(cfg, "seed", 42),
        )
    }

    # build model (CPU)
    decoder_backend = str(getattr(cfg, "decoder_backend", "lookup"))
    decoder_cfg = {
        "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
        "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
        "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
        "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
    }
    core = FCCAutoEncoder(
        device="cpu",
        grid_res=int(getattr(cfg, "grid_res", 1000)),
        decoder_backend=decoder_backend,
        decoder_config=decoder_cfg,
    ).to("cpu")
    core.eval()

    if not ORIX_AVAILABLE:
        raise RuntimeError("ORIX is required for IPF rendering; install orix or set up the environment.")
    laue = getattr(orix_sym, "Oh")

    val_iter = iter(loaders["Test"])
    for sidx, batch in enumerate(val_iter):
        _, hr = batch
        hr0 = hr[0]  # (4,H,W)
        H, W = int(hr0.shape[1]), int(hr0.shape[2])
        q_all = hr0.permute(1, 2, 0).reshape(-1, 4).to(torch.float32)  # (N,4)
        # Ensure we have a WXYZ tensor for model and IPF rendering
        if INPUT_ORDER.lower() == "wxyz":
            q_all_wxyz = q_all
        else:
            # convert XYZW -> WXYZ
            q_xyzw_np = q_all.cpu().numpy().reshape(H, W, 4)
            q_wxyz_np = xyzw_to_wxyz(q_xyzw_np).reshape(-1, 4)
            q_all_wxyz = torch.from_numpy(q_wxyz_np).to(torch.float32)
        N = q_all.shape[0]
        print(f"Processing validation sample {sidx}: {H}x{W}, N={N}")

        # compute decoded FZ orientations and canonical (orig->FZ) using chunks
        chunk = 65536
        bad_entries = []
        with torch.no_grad():
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                q_batch = q_all_wxyz[start:end].to("cpu")
                q_dec_fz = core(q_batch, normalize_input=True).cpu().numpy()  # (M,4)
                q_orig_fz = core.reduce_to_fz(q_batch).cpu().numpy()  # (M,4)
                # compute errors (radians)
                w = np.sum(q_orig_fz * q_dec_fz, axis=-1)
                w_clamped = np.clip(np.abs(w), -1.0, 1.0)
                errs = 2.0 * np.arccos(w_clamped)  # (M,)
                mis_deg = errs * 180.0 / math.pi

                for i in range(end - start):
                    idx = start + i
                    err = float(errs[i])
                    if err > THRESH_RAD:
                        row, col = divmod(idx, W)
                        bad_entries.append(
                            {
                                "index": int(idx),
                                "row": int(row),
                                "col": int(col),
                                "error_rad": float(err),
                                "misorientation_deg": float(mis_deg[i]),
                                "q_orig_wxyz": " ".join(map(str, q_orig_fz[i].tolist())),
                                "q_dec_wxyz": " ".join(map(str, q_dec_fz[i].tolist())),
                            }
                        )

        # output dir for this sample
        sample_out = OUT_DIR / f"sample_{sidx:03d}"
        sample_out.mkdir(parents=True, exist_ok=True)

        # write CSV for this sample
        csv_path = sample_out / "bad_pixels.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "row",
                    "col",
                    "error_rad",
                    "misorientation_deg",
                    "q_orig_wxyz",
                    "q_dec_wxyz",
                ]
            )
            for rec in bad_entries:
                writer.writerow(
                    [
                        rec["index"],
                        rec["row"],
                        rec["col"],
                        f"{rec['error_rad']:.6e}",
                        f"{rec['misorientation_deg']:.6f}",
                        rec["q_orig_wxyz"],
                        rec["q_dec_wxyz"],
                    ]
                )
        print("Wrote CSV:", csv_path, "found", len(bad_entries), "bad pixels")

        # create overlay image (IPF coloring), mark bad pixels with red circles
        overlay_path = sample_out / "bad_pixels_overlay.png"
        try:
            q_wxyz_full = q_all_wxyz.cpu().numpy().reshape(H, W, 4)
            orig_ipf = ipf_rgb_orix(q_wxyz_full, laue, ref_dir="Z")
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            ax.imshow(orig_ipf)

            if bad_entries:
                xs = [r["col"] for r in bad_entries]
                ys = [r["row"] for r in bad_entries]
                ax.scatter(xs, ys, s=40, facecolors="none", edgecolors="r", linewidths=1.2)
            ax.set_title("Bad pixels overlay (red markers)")
            ax.axis("off")
            fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print("Saved overlay:", overlay_path)
        except Exception as exc:
            print("Failed to create IPF overlay image:", exc)

        # create thumbnails per bad pixel: original-FZ (left) vs decoded (right)
        # For thumbnails we need q_orig_fz and q_dec_fz again for local patches; recompute per bad pixel index
        if bad_entries:
            # convert q_all to numpy for easy slicing
            q_all_np = q_all.cpu().numpy()
            # build a map of indices -> decoded/quants by processing chunks again and storing only needed indices
            needed_idx = sorted([b["index"] for b in bad_entries[:MAX_THUMBS]])
            idx_set = set(needed_idx)
            decoded_map = {}
            orig_map = {}
            with torch.no_grad():
                for start in range(0, N, chunk):
                    end = min(start + chunk, N)
                    q_batch = q_all[start:end].to("cpu")
                    q_dec_fz = core(q_batch, normalize_input=True).cpu().numpy()
                    q_orig_fz = core.reduce_to_fz(q_batch).cpu().numpy()
                    for i in range(end - start):
                        idx = start + i
                        if idx in idx_set:
                            decoded_map[idx] = q_dec_fz[i]
                            orig_map[idx] = q_orig_fz[i]
                    if len(decoded_map) == len(needed_idx):
                        break

            for rec in bad_entries[:MAX_THUMBS]:
                idx = rec["index"]
                row = rec["row"]
                col = rec["col"]
                q_o = orig_map.get(idx)
                q_d = decoded_map.get(idx)
                if q_o is None or q_d is None:
                    continue
                # build small patch centered on pixel (use available q_all for neighbor pixels)
                r0 = max(0, row - PATCH_HALF)
                r1 = min(H, row + PATCH_HALF + 1)
                c0 = max(0, col - PATCH_HALF)
                c1 = min(W, col + PATCH_HALF + 1)
                patch = q_all_np.reshape(H, W, 4)[r0:r1, c0:c1, :]
                # convert neighbor orientations to FZ via the model then IPF-render
                try:
                    # patch is XYZW; convert to WXYZ for reduce_to_fz
                    patch_arr = patch.reshape(-1, 4)
                    if INPUT_ORDER.lower() == "wxyz":
                        patch_wxyz = patch_arr
                    else:
                        patch_wxyz = xyzw_to_wxyz(patch_arr)
                    with torch.no_grad():
                        tq = torch.from_numpy(patch_wxyz.astype(np.float32))
                        patch_fz = core.reduce_to_fz(tq).cpu().numpy()
                    patch_orig_rgb = ipf_rgb_orix(patch_fz.reshape((r1 - r0), (c1 - c0), 4), laue, ref_dir="Z")
                except Exception as exc:
                    print(f"Failed to render patch IPF for idx {idx}:", exc)
                    patch_orig_rgb = np.zeros(((r1 - r0), (c1 - c0), 3), dtype=np.uint8)

                # center colors (orig-FZ and decoded-FZ) are available in orig_map/decoded_map as WXYZ
                center_rgb_orig = ipf_rgb_orix(orig_map[idx].reshape(1, 1, 4), laue, ref_dir="Z")[0, 0]
                center_rgb_dec = ipf_rgb_orix(decoded_map[idx].reshape(1, 1, 4), laue, ref_dir="Z")[0, 0]

                # create figure: left full patch with orig-FZ colors, right two small swatches (orig vs dec)
                fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
                axes[0].imshow((patch_orig_rgb * 255).astype(np.uint8))
                axes[0].scatter([col - c0], [row - r0], s=80, facecolors="none", edgecolors="r")
                axes[0].set_title(f"Patch @{row},{col}")
                axes[0].axis("off")

                # right: two swatches
                sw_img = np.zeros((50, 100, 3), dtype=np.uint8)
                sw_img[:, :50, :] = (center_rgb_orig * 255).astype(np.uint8)
                sw_img[:, 50:, :] = (center_rgb_dec * 255).astype(np.uint8)
                axes[1].imshow(sw_img)
                axes[1].set_title("orig-FZ (L) | decoded (R)")
                axes[1].axis("off")

                thumb_path = sample_out / f"bad_{idx}_{row}_{col}.png"
                fig.savefig(thumb_path, bbox_inches="tight", pad_inches=0.05)
                plt.close(fig)

            print(f"Saved thumbnails for bad pixels (first {MAX_THUMBS}) in {sample_out}.")

    print("Done processing validation set.")


if __name__ == "__main__":
    main()