from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SR_4x4_from_4x1_ocrp_anchorless import (
    FeatureBankClusterer,
    LocalIsoCrystalEncoder,
    QuaternionBankClusterer,
    _build_local_patch_bank,
    _left_mult_matrix_wxyz_batch,
    _misorientation_angle_sym,
    _normalize_quaternions,
)


def _edge_index(h: int, w: int, connectivity: int) -> tuple[torch.Tensor, torch.Tensor]:
    src: list[int] = []
    dst: list[int] = []
    for y in range(h):
        for x in range(w):
            i = y * w + x
            if x + 1 < w:
                src.append(i)
                dst.append(i + 1)
            if y + 1 < h:
                src.append(i)
                dst.append(i + w)
            if connectivity == 8 and y + 1 < h:
                if x + 1 < w:
                    src.append(i)
                    dst.append(i + w + 1)
                if x - 1 >= 0:
                    src.append(i)
                    dst.append(i + w - 1)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def _quantiles(x: torch.Tensor, probs: list[float]) -> dict[str, float]:
    if x.numel() == 0:
        return {f"q{int(p * 100):02d}": float("nan") for p in probs}
    q = torch.quantile(x.to(torch.float64), torch.tensor(probs, dtype=torch.float64))
    return {f"q{int(p * 100):02d}": float(v) for p, v in zip(probs, q)}


def _summarize(name: str, values: torch.Tensor) -> dict[str, float]:
    out = {
        f"{name}_mean": float(values.mean().item()) if values.numel() else float("nan"),
        f"{name}_max": float(values.max().item()) if values.numel() else float("nan"),
    }
    out.update({f"{name}_{k}": v for k, v in _quantiles(values, [0.5, 0.9, 0.95, 0.99]).items()})
    return out


@torch.no_grad()
def diagnose_material(
    *,
    material: str,
    run_dir: Path,
    max_samples: int | None,
    chunk_quats: int,
    chunk_edges: int,
    cluster_compare_samples: int,
) -> dict[str, object]:
    summary_path = run_dir / "inference/test_best/summary.json"
    config_path = run_dir / "config_new.json"
    summary = json.loads(summary_path.read_text())
    cfg = json.loads(config_path.read_text())

    records = summary["records"]
    if max_samples is not None:
        records = records[: int(max_samples)]

    encoder = LocalIsoCrystalEncoder(
        crystal=str(cfg.get("crystal", "fcc")),
        d6_convention=str(cfg.get("d6_convention", "z_axis")),
        embedding_mode=str(cfg.get("embedding_mode", "direct_reynolds")),
        max_harmonic_l=cfg.get("max_harmonic_l"),
        embedding_metric_calibration=cfg.get("embedding_metric_calibration", "none"),
        dtype=torch.float32,
        device="cpu",
    ).eval()

    threshold_deg = float(cfg.get("cluster_threshold_deg", 2.0))
    threshold_rad = float(np.deg2rad(threshold_deg))
    threshold_l2 = float(
        cfg.get("cluster_feature_l2_threshold")
        if cfg.get("cluster_feature_l2_threshold") is not None
        else threshold_rad
    )
    feature_irreps = str(cfg.get("feature_irreps", "full")).lower()
    connectivity = int(cfg.get("cluster_connectivity", 8))
    window_size = int(cfg.get("window_size", 9))

    sym_ops_mat = _left_mult_matrix_wxyz_batch(_normalize_quaternions(encoder.sym_ops.detach().cpu()))
    quat_clusterer = QuaternionBankClusterer(
        sym_ops_quat=encoder.sym_ops.detach().cpu(),
        threshold_deg=threshold_deg,
        connectivity=connectivity,
        window_size=window_size,
    )
    feature_clusterer = FeatureBankClusterer(
        threshold_l2=threshold_l2,
        connectivity=connectivity,
        window_size=window_size,
    )

    total_edges = 0
    quat_keep_total = 0
    feat_keep_total = 0
    agree_total = 0
    feat_keep_quat_reject_total = 0
    quat_keep_feat_reject_total = 0

    all_small_l2: list[torch.Tensor] = []
    all_small_deg: list[torch.Tensor] = []
    feat_keep_deg: list[torch.Tensor] = []
    quat_keep_l2: list[torch.Tensor] = []
    feat_reject_but_quat_keep_deg: list[torch.Tensor] = []
    feat_keep_but_quat_reject_deg: list[torch.Tensor] = []

    sample_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    for rec in records:
        q_np = np.load(rec["lr_npy"])
        if q_np.ndim == 3:
            h, w = int(q_np.shape[0]), int(q_np.shape[1])
            q = torch.from_numpy(q_np.reshape(h * w, 4)).to(torch.float32)
        elif q_np.ndim == 2:
            n = int(q_np.shape[0])
            h, w = map(int, rec["lr_shape"])
            if h * w != n:
                raise ValueError(f"{rec['lr_npy']} has N={n}, lr_shape={rec['lr_shape']}")
            q = torch.from_numpy(q_np).to(torch.float32)
        else:
            raise ValueError(f"Unexpected LR quaternion shape {q_np.shape} in {rec['lr_npy']}")
        q = _normalize_quaternions(q)

        feat_chunks = []
        for start in range(0, int(q.shape[0]), int(chunk_quats)):
            q_chunk = q[start : start + int(chunk_quats)]
            if feature_irreps == "a1":
                feat_chunks.append(encoder.forward_a1(q_chunk).detach().cpu())
            else:
                feat_chunks.append(encoder.forward_full(q_chunk).detach().cpu())
        f = torch.cat(feat_chunks, dim=0).to(torch.float32)

        edge_a, edge_b = _edge_index(h, w, connectivity)
        sample_edges = int(edge_a.numel())
        sample_quat_keep = 0
        sample_feat_keep = 0
        sample_agree = 0
        sample_feat_keep_quat_reject = 0
        sample_quat_keep_feat_reject = 0

        for start in range(0, sample_edges, int(chunk_edges)):
            a = edge_a[start : start + int(chunk_edges)]
            b = edge_b[start : start + int(chunk_edges)]
            deg = torch.rad2deg(_misorientation_angle_sym(q.index_select(0, a), q.index_select(0, b), sym_ops_mat))
            l2 = (f.index_select(0, a) - f.index_select(0, b)).pow(2).sum(dim=-1).sqrt()

            quat_keep = deg <= threshold_deg
            feat_keep = l2 <= threshold_l2
            sample_quat_keep += int(quat_keep.sum().item())
            sample_feat_keep += int(feat_keep.sum().item())
            sample_agree += int((quat_keep == feat_keep).sum().item())
            sample_feat_keep_quat_reject += int((feat_keep & ~quat_keep).sum().item())
            sample_quat_keep_feat_reject += int((quat_keep & ~feat_keep).sum().item())

            small = deg <= 5.0
            if bool(small.any().item()):
                all_small_l2.append(l2[small].cpu())
                all_small_deg.append(deg[small].cpu())
            if bool(feat_keep.any().item()):
                feat_keep_deg.append(deg[feat_keep].cpu())
            if bool(quat_keep.any().item()):
                quat_keep_l2.append(l2[quat_keep].cpu())
            miss = quat_keep & ~feat_keep
            extra = feat_keep & ~quat_keep
            if bool(miss.any().item()):
                feat_reject_but_quat_keep_deg.append(deg[miss].cpu())
            if bool(extra.any().item()):
                feat_keep_but_quat_reject_deg.append(deg[extra].cpu())

        total_edges += sample_edges
        quat_keep_total += sample_quat_keep
        feat_keep_total += sample_feat_keep
        agree_total += sample_agree
        feat_keep_quat_reject_total += sample_feat_keep_quat_reject
        quat_keep_feat_reject_total += sample_quat_keep_feat_reject
        if len(sample_rows) < 5:
            sample_rows.append(
                {
                    "sample_id": int(rec.get("sample_id", len(sample_rows))),
                    "edges": sample_edges,
                    "quat_keep_frac": sample_quat_keep / sample_edges,
                    "feature_keep_frac": sample_feat_keep / sample_edges,
                    "agreement_frac": sample_agree / sample_edges,
                    "feature_extra_edges": sample_feat_keep_quat_reject,
                    "feature_missed_quat_edges": sample_quat_keep_feat_reject,
                }
            )
        if len(cluster_rows) < int(cluster_compare_samples):
            bank_q = _build_local_patch_bank(q, img_shape=(h, w), window_size=window_size)
            bank_f = _build_local_patch_bank(f, img_shape=(h, w), window_size=window_size)
            q_ids = quat_clusterer(bank_q)
            f_ids = feature_clusterer(bank_f)
            same_nodes = q_ids == f_ids
            same_windows = same_nodes.all(dim=-1)
            q_cluster_counts = torch.tensor(
                [torch.unique(row).numel() for row in q_ids.reshape(-1, q_ids.shape[-1])],
                dtype=torch.float32,
            )
            f_cluster_counts = torch.tensor(
                [torch.unique(row).numel() for row in f_ids.reshape(-1, f_ids.shape[-1])],
                dtype=torch.float32,
            )
            cluster_rows.append(
                {
                    "sample_id": int(rec.get("sample_id", len(cluster_rows))),
                    "windows": int(q_ids.shape[0]),
                    "node_label_agreement_frac": float(same_nodes.to(torch.float32).mean().item()),
                    "exact_window_label_agreement_frac": float(same_windows.to(torch.float32).mean().item()),
                    "quaternion_cluster_count_mean": float(q_cluster_counts.mean().item()),
                    "feature_cluster_count_mean": float(f_cluster_counts.mean().item()),
                    "quaternion_cluster_count_max": int(q_cluster_counts.max().item()),
                    "feature_cluster_count_max": int(f_cluster_counts.max().item()),
                }
            )

    small_l2 = torch.cat(all_small_l2) if all_small_l2 else torch.empty(0)
    small_deg = torch.cat(all_small_deg) if all_small_deg else torch.empty(0)
    ratio = small_l2 / torch.deg2rad(small_deg).clamp_min(1e-8)
    feat_keep_deg_t = torch.cat(feat_keep_deg) if feat_keep_deg else torch.empty(0)
    quat_keep_l2_t = torch.cat(quat_keep_l2) if quat_keep_l2 else torch.empty(0)
    miss_deg_t = (
        torch.cat(feat_reject_but_quat_keep_deg)
        if feat_reject_but_quat_keep_deg
        else torch.empty(0)
    )
    extra_deg_t = (
        torch.cat(feat_keep_but_quat_reject_deg)
        if feat_keep_but_quat_reject_deg
        else torch.empty(0)
    )

    return {
        "material": material,
        "run_dir": str(run_dir),
        "samples": len(records),
        "total_unique_lr_edges": total_edges,
        "feature_irreps": feature_irreps,
        "embedding_mode": cfg.get("embedding_mode"),
        "embedding_metric_calibration": cfg.get("embedding_metric_calibration"),
        "max_harmonic_l": cfg.get("max_harmonic_l"),
        "threshold_deg": threshold_deg,
        "threshold_rad": threshold_rad,
        "threshold_l2": threshold_l2,
        "connectivity": connectivity,
        "quaternion_keep_frac": quat_keep_total / total_edges,
        "feature_keep_frac": feat_keep_total / total_edges,
        "decision_agreement_frac": agree_total / total_edges,
        "feature_extra_vs_quat_frac": feat_keep_quat_reject_total / total_edges,
        "feature_missed_quat_frac": quat_keep_feat_reject_total / total_edges,
        "counts": {
            "quat_keep": quat_keep_total,
            "feature_keep": feat_keep_total,
            "agree": agree_total,
            "feature_extra_vs_quat": feat_keep_quat_reject_total,
            "feature_missed_quat": quat_keep_feat_reject_total,
        },
        "small_angle_l2_over_rad": _summarize("ratio", ratio),
        "small_angle_deg": _summarize("deg", small_deg),
        "feature_kept_edge_deg": _summarize("deg", feat_keep_deg_t),
        "quaternion_kept_edge_l2": _summarize("l2", quat_keep_l2_t),
        "missed_quat_edge_deg": _summarize("deg", miss_deg_t),
        "extra_feature_edge_deg": _summarize("deg", extra_deg_t),
        "first_samples": sample_rows,
        "cluster_label_compare": cluster_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-quats", type=int, default=65536)
    parser.add_argument("--chunk-edges", type=int, default=262144)
    parser.add_argument("--cluster-compare-samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    runs = [
        (
            "IN718",
            root
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42",
        ),
        (
            "Ti_Al_1pct",
            root
            / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42",
        ),
    ]
    results = [
        diagnose_material(
            material=material,
            run_dir=run_dir,
            max_samples=args.max_samples,
            chunk_quats=args.chunk_quats,
            chunk_edges=args.chunk_edges,
            cluster_compare_samples=args.cluster_compare_samples,
        )
        for material, run_dir in runs
    ]
    text = json.dumps(results, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
