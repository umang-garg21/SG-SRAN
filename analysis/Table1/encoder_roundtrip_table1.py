"""Notebook-friendly helpers for Table 1 encoder round-trip fidelity."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# `orix` pulls in numba-cached helpers during import. In this environment we
# need an explicit writable cache dir or the import can fail before the notebook
# even starts running.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    dataset_root: str | Path
    crystal: str
    d6_convention: str = "z_axis"
    model_module: str = "models.SR_4x4_from_4x1_ocrp_anchorless"
    split: str = "Test"
    embedding_mode: str | None = "direct_reynolds"
    max_harmonic_l: int | None = None
    embedding_metric_calibration: object = "isometric"
    target_irreps: str = "full"
    decoder_cubochoric_resolution: int = 1
    decoder_method: str = "cubochoric"
    decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables"


@dataclass(frozen=True)
class DecoderVariant:
    label: str
    num_starts: int
    steps: int
    lr: float = 0.03


ROUNDTRIP_FLOW = [
    "load one HR test scan",
    "flatten the scan to passive quaternions with shape (N, 4)",
    "encode quaternions with LocalIsoCrystalEncoder.forward_full(...) unless target_irreps='a1'",
    "decode features with CubochoricOptimizingLocalIsoDecoder(...)",
    "compute symmetry-aware dS(q_decoded, q_ref) in radians",
    "average dS within each scan, then report mean +/- std across scans",
]


def default_table1_datasets() -> list[DatasetSpec]:
    """Return the repo-local dataset specs for the paper's Table 1."""
    return [
        DatasetSpec(
            label="IN718 (FCC)",
            dataset_root="/data/home/umang/Materials/Materials_data_mount/datasets/IN718_QSR_x4",
            crystal="fcc",
            max_harmonic_l=4,
            decoder_table_cache_dir="out/decoder_lookup_tables/direct_reynolds_fcc_l4_full",
        ),
        DatasetSpec(
            label="Ti-6Al-4V (HCP)",
            dataset_root="/data/home/umang/Materials/Materials_data_mount/datasets/Ti_Al_1pct_QSR_x4",
            crystal="hcp",
            d6_convention="z_axis",
            max_harmonic_l=6,
            decoder_table_cache_dir="out/decoder_lookup_tables/direct_reynolds_hcp_l6_full",
        ),
    ]


def default_table1_variants() -> list[DecoderVariant]:
    """Return the two decoder settings described in Table 1."""
    return [
        DecoderVariant(
            label="LUT lookup only (no refinement)",
            num_starts=1,
            steps=0,
            lr=0.03,
        ),
        DecoderVariant(
            label="LUT + local refinement (12 steps, 8 starts)",
            num_starts=8,
            steps=12,
            lr=0.03,
        ),
    ]


def _resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def describe_roundtrip_flow() -> list[str]:
    """Return the exact evaluation flow used for this notebook/task."""
    return list(ROUNDTRIP_FLOW)


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    return quats / norm


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def _flatten_quaternion_image(quats: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if quats.dim() != 3:
        raise ValueError(f"Expected a rank-3 quaternion image, got {tuple(quats.shape)}")
    if int(quats.shape[0]) == 4:
        _, h, w = quats.shape
        return quats.permute(1, 2, 0).reshape(h * w, 4), (int(h), int(w))
    if int(quats.shape[-1]) == 4:
        h, w, _ = quats.shape
        return quats.reshape(h * w, 4), (int(h), int(w))
    raise ValueError(f"Expected quaternion axis of size 4, got {tuple(quats.shape)}")


def _load_dataset_info(dataset_root: Path) -> dict[str, object]:
    info_path = dataset_root / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_hr_files(
    dataset_root: Path,
    split: str,
    take_first: int | None = None,
) -> list[Path]:
    info = _load_dataset_info(dataset_root)
    split_name = str(split).capitalize()
    split_info = info["splits"][split_name]
    hr_glob = Path(str(split_info["HR_glob"]))
    hr_files = sorted(hr_glob.parent.glob(hr_glob.name))
    if not hr_files:
        hr_files = sorted((dataset_root / split_name / "HR_Data").glob("*.npy"))
    if not hr_files:
        raise RuntimeError(f"No HR files found for split={split_name} in {dataset_root}")
    if take_first is not None:
        hr_files = hr_files[: int(take_first)]
    return hr_files


def symmetry_aware_error_rad(
    q_pred: torch.Tensor,
    q_ref: torch.Tensor,
    sym_ops_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetry-aware angular error for passive quaternions:
        dS(q_pred, q_ref) = min_s 2 * acos(|<q_pred, s^{-1} ⊗ q_ref>|)
    """
    q_pred = _normalize_quaternions(q_pred.to(torch.float32))
    q_ref = _normalize_quaternions(q_ref.to(torch.float32))
    sym_ops_inv = sym_ops_inv.to(device=q_ref.device, dtype=torch.float32)

    n = int(q_ref.shape[0])
    g = int(sym_ops_inv.shape[0])
    orbit = _quat_mul(
        sym_ops_inv.unsqueeze(0).expand(n, -1, -1).reshape(n * g, 4),
        q_ref.unsqueeze(1).expand(-1, g, -1).reshape(n * g, 4),
    ).reshape(n, g, 4)
    orbit = _normalize_quaternions(orbit.reshape(n * g, 4)).reshape(n, g, 4)

    dots = (q_pred.unsqueeze(1) * orbit).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dots).min(dim=1).values


def _build_encoder_decoder(
    spec: DatasetSpec,
    variant: DecoderVariant,
    device: str | torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]:
    module = importlib.import_module(spec.model_module)
    encoder_cls = getattr(module, "LocalIsoCrystalEncoder")
    decoder_cls = getattr(module, "CubochoricOptimizingLocalIsoDecoder")

    encoder_kwargs = {
        "crystal": spec.crystal,
        "d6_convention": spec.d6_convention,
        "dtype": torch.float32,
        "device": device,
    }
    if spec.embedding_mode is not None:
        encoder_kwargs.update(
            {
                "embedding_mode": spec.embedding_mode,
                "max_harmonic_l": spec.max_harmonic_l,
                "embedding_metric_calibration": spec.embedding_metric_calibration,
            }
        )

    encoder = encoder_cls(**encoder_kwargs).eval()

    decoder = decoder_cls(
        encoder=encoder,
        cubochoric_resolution=int(spec.decoder_cubochoric_resolution),
        method=str(spec.decoder_method),
        num_starts=int(variant.num_starts),
        steps=int(variant.steps),
        lr=float(variant.lr),
        target_irreps=str(spec.target_irreps),
        table_cache_dir=_resolve_path(spec.decoder_table_cache_dir),
    ).eval()
    return encoder, decoder, encoder.sym_ops_inv


def _encode_for_spec(
    encoder: torch.nn.Module,
    spec: DatasetSpec,
    quats_passive: torch.Tensor,
) -> torch.Tensor:
    if str(spec.target_irreps).lower() == "a1":
        return encoder.forward_a1(quats_passive)
    return encoder.forward_full(quats_passive)


def evaluate_roundtrip(
    spec: DatasetSpec,
    variant: DecoderVariant,
    *,
    device: str | torch.device | None = None,
    take_first: int | None = None,
    chunk_size: int = 4096,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate one dataset/variant pair and return per-scan metrics."""
    dataset_root = Path(spec.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    hr_files = _list_hr_files(dataset_root, spec.split, take_first=take_first)
    encoder, decoder, sym_ops_inv = _build_encoder_decoder(spec, variant, device=device)

    rows: list[dict[str, object]] = []
    iterator: Iterable[tuple[int, Path]] = enumerate(hr_files)
    if progress:
        iterator = tqdm(
            iterator,
            total=len(hr_files),
            desc=f"{spec.label} | {variant.label}",
        )

    for idx, hr_path in iterator:
        hr_np = np.load(hr_path, mmap_mode="r")
        hr_np = np.array(hr_np, dtype=np.float32, copy=True)
        hr = torch.from_numpy(hr_np)
        q_ref_flat, (height, width) = _flatten_quaternion_image(hr)

        err_chunks: list[torch.Tensor] = []
        for start in range(0, int(q_ref_flat.shape[0]), int(chunk_size)):
            end = min(start + int(chunk_size), int(q_ref_flat.shape[0]))
            q_ref = q_ref_flat[start:end].to(device=device, dtype=torch.float32)
            q_ref = _normalize_quaternions(q_ref)

            with torch.no_grad():
                feat = _encode_for_spec(encoder, spec, q_ref)
            with torch.enable_grad():
                q_pred = decoder(feat)
            with torch.no_grad():
                err_chunks.append(
                    symmetry_aware_error_rad(q_pred.detach(), q_ref, sym_ops_inv).cpu()
                )

        d_s = torch.cat(err_chunks, dim=0)
        rows.append(
            {
                "dataset_label": spec.label,
                "variant_label": variant.label,
                "dataset_root": str(dataset_root),
                "scan_index": int(idx),
                "scan_file": hr_path.name,
                "hr_path": str(hr_path),
                "height": int(height),
                "width": int(width),
                "num_pixels": int(d_s.numel()),
                "scan_mean_dS_rad": float(d_s.mean().item()),
                "scan_std_dS_rad": float(d_s.std(unbiased=False).item()),
                "scan_max_dS_rad": float(d_s.max().item()),
            }
        )

    return pd.DataFrame(rows)


def summarize_scan_metrics(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scan-level means into the table-ready statistics."""
    if scan_df.empty:
        raise ValueError("scan_df is empty")

    summary_rows: list[dict[str, object]] = []
    grouped = scan_df.groupby(["dataset_label", "variant_label"], sort=False)
    for (dataset_label, variant_label), group in grouped:
        scan_means = group["scan_mean_dS_rad"]
        num_pixels = group["num_pixels"]
        weighted_mean = float((scan_means * num_pixels).sum() / num_pixels.sum())
        std_scan_mean = float(scan_means.std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(
            {
                "dataset_label": dataset_label,
                "variant_label": variant_label,
                "num_scans": int(len(group)),
                "mean_of_scan_mean_dS_rad": float(scan_means.mean()),
                "std_of_scan_mean_dS_rad": std_scan_mean,
                "weighted_mean_dS_rad": weighted_mean,
                "paper_value": f"{float(scan_means.mean()):.4f} +/- {std_scan_mean:.4f}",
            }
        )
    return pd.DataFrame(summary_rows)


def pivot_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the summary into the paper's 2-column layout."""
    table = summary_df.pivot(
        index="variant_label",
        columns="dataset_label",
        values="paper_value",
    )
    return table.reset_index()


def evaluate_default_table1(
    *,
    device: str | torch.device | None = None,
    take_first: int | None = None,
    chunk_size: int = 4096,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both paper datasets and both decoder variants."""
    all_scan_frames = []
    for spec in default_table1_datasets():
        for variant in default_table1_variants():
            all_scan_frames.append(
                evaluate_roundtrip(
                    spec,
                    variant,
                    device=device,
                    take_first=take_first,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            )

    scan_df = pd.concat(all_scan_frames, ignore_index=True)
    summary_df = summarize_scan_metrics(scan_df)
    table_df = pivot_paper_table(summary_df)
    return scan_df, summary_df, table_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Table 1 encoder round-trip metrics.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--take-first", type=int, default=None, help="Optional smoke-test scan count.")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--out-dir",
        default="analysis/Table1/out/table1_roundtrip_direct_reynolds",
        help="Output directory for scan_metrics.csv, summary_metrics.csv and paper_table.csv.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    scan_df, summary_df, table_df = evaluate_default_table1(
        device=args.device,
        take_first=args.take_first,
        chunk_size=args.chunk_size,
        progress=not args.no_progress,
    )

    out_dir = _resolve_path(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(out_dir / "scan_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    table_df.to_csv(out_dir / "paper_table.csv", index=False)
    print(table_df.to_string(index=False))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
