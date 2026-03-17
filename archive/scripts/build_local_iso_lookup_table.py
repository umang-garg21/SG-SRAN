import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from local_iso_embedding import (  # noqa: E402
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
    return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)


def _passive_to_active_quats(quats_passive: torch.Tensor) -> torch.Tensor:
    # Passive <-> active conversion for unit quaternions is conjugation.
    return _normalize_quaternions(_quat_conjugate(quats_passive))


def _sample_fz_quaternions_passive(
    group_name: str,
    resolution: int,
    method: str,
    dtype: torch.dtype,
    device: str | torch.device,
    max_rows: int | None = None,
) -> torch.Tensor:
    from orix.quaternion import symmetry
    from orix.sampling import get_sample_fundamental

    g = str(group_name).upper()
    if g == "O":
        point_group = symmetry.Oh
    elif g == "D6":
        point_group = symmetry.D6h
    else:
        raise ValueError(f"group_name must be 'O' or 'D6', got {group_name}")

    rot = get_sample_fundamental(
        int(resolution),
        point_group=point_group,
        method=str(method),
    )

    raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
    if raw.ndim != 2:
        raw = raw.reshape(-1, 4)
    if raw.shape[-1] != 4 and raw.shape[0] == 4:
        raw = raw.T
    if raw.shape[-1] != 4:
        raise ValueError(f"Unexpected quaternion shape from orix: {tuple(raw.shape)}")

    q_passive = torch.as_tensor(raw, dtype=dtype, device=device)
    q_passive = _normalize_quaternions(q_passive)

    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0:
            q_passive = q_passive[:max_rows]

    return q_passive


@torch.no_grad()
def build_local_iso_lookup_table(
    group_name: str,
    out_path: str | Path,
    resolution: int = 1,
    method: str = "cubochoric",
    feature_space: str = "irreps",
    d6_convention: str = "z_axis",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 32768,
    max_rows: int | None = None,
    store_quats_convention: str = "passive",
) -> dict[str, object]:
    """
    Build and save packed lookup table:
      [ q_w, q_x, q_y, q_z, feat..., feat_norm_sq ]

    Convention handling:
    - ORIX FZ samples are treated as passive quaternions.
    - Irreps features use model.forward_irreps_passive(passive_quats).
    - Raw features convert passive -> active before evaluation.
    - Stored quaternion columns can be passive (default) or active.
    """
    group_name = str(group_name).upper()
    feature_space = str(feature_space).lower()
    store_quats_convention = str(store_quats_convention).lower()

    if feature_space not in {"irreps", "raw"}:
        raise ValueError(
            f"feature_space must be 'irreps' or 'raw', got {feature_space}"
        )
    if store_quats_convention not in {"passive", "active"}:
        raise ValueError(
            "store_quats_convention must be 'passive' or 'active', "
            f"got {store_quats_convention}"
        )

    device = torch.device(device)
    chunk_size = max(1024, int(chunk_size))

    if group_name == "O":
        model = build_local_iso_fcc_embedding(dtype=dtype, device=device).eval()
    elif group_name == "D6":
        model = build_local_iso_hcp_embedding(
            d6_convention=d6_convention,
            dtype=dtype,
            device=device,
        ).eval()
    else:
        raise ValueError(f"group_name must be 'O' or 'D6', got {group_name}")

    quats_passive = _sample_fz_quaternions_passive(
        group_name=group_name,
        resolution=resolution,
        method=method,
        dtype=dtype,
        device=device,
        max_rows=max_rows,
    )

    n_rows = int(quats_passive.shape[0])
    if n_rows == 0:
        raise ValueError("No quaternions sampled. Increase resolution or max_rows.")

    if feature_space == "irreps":
        probe = model.forward_irreps_passive(quats_passive[:1])
    else:
        probe = model.forward_from_quaternions(
            _passive_to_active_quats(quats_passive[:1]),
            raw=False,
        )
    feat_dim = int(probe.shape[-1])

    packed = np.empty((n_rows, 4 + feat_dim + 1), dtype=np.float32)
    t0 = time.perf_counter()

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        q_passive_chunk = quats_passive[start:end]

        if feature_space == "irreps":
            feat = model.forward_irreps_passive(q_passive_chunk)
            q_active_chunk = (
                _passive_to_active_quats(q_passive_chunk)
                if store_quats_convention == "active"
                else None
            )
        else:
            q_active_chunk = _passive_to_active_quats(q_passive_chunk)
            feat = model.forward_from_quaternions(q_active_chunk, raw=True)

        feat_norm = (feat * feat).sum(dim=-1, keepdim=True)

        if store_quats_convention == "passive":
            q_store = q_passive_chunk
        else:
            q_store = q_active_chunk

        packed[start:end, :4] = (
            q_store.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        packed[start:end, 4 : 4 + feat_dim] = (
            feat.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        packed[start:end, 4 + feat_dim :] = (
            feat_norm.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        if start == 0 or end == n_rows or ((start // chunk_size) % 20 == 0):
            elapsed = time.perf_counter() - t0
            rows_per_s = end / max(1e-6, elapsed)
            print(
                f"[{group_name}] rows {end}/{n_rows} ({100.0 * end / n_rows:.1f}%) "
                f"speed={rows_per_s:.1f} rows/s"
            )

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), packed)

    dt = time.perf_counter() - t0
    meta = {
        "path": str(out_path),
        "shape": tuple(packed.shape),
        "group_name": group_name,
        "feature_space": feature_space,
        "resolution": int(resolution),
        "method": str(method),
        "d6_convention": str(d6_convention),
        "device": str(device),
        "dtype": str(dtype),
        "seconds": float(dt),
        "rows_per_sec": float(n_rows / max(1e-6, dt)),
        "feature_dim": int(feat_dim),
        "sample_quat_convention": "passive",
        "embedding_input_convention": "active",
        "feature_eval_api": (
            "forward_irreps_passive"
            if feature_space == "irreps"
            else "forward_from_quaternions(raw=True)"
        ),
        "stored_quat_convention": store_quats_convention,
    }
    print(
        f"saved {out_path} shape={packed.shape} feature_dim={feat_dim} "
        f"stored_q={store_quats_convention} time={dt:.2f}s"
    )
    return meta


def _default_output_path(
    group_name: str,
    resolution: int,
    feature_space: str,
    d6_convention: str,
) -> Path:
    g = str(group_name).upper()
    feature_space = str(feature_space).lower()
    if g == "D6":
        name = (
            f"local_iso_lookup_{g}_{d6_convention}_res{int(resolution)}_"
            f"{feature_space}.npy"
        )
    else:
        name = f"local_iso_lookup_{g}_res{int(resolution)}_{feature_space}.npy"
    return (PROJECT_ROOT / "symmetry_groups" / name).resolve()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    jobs = [
        {
            "group_name": "O",
            "resolution": 1,
            "method": "cubochoric",
            "feature_space": "irreps",
            "d6_convention": "z_axis",
            "max_rows": None,
        },
        {
            "group_name": "D6",
            "resolution": 1,
            "method": "cubochoric",
            "feature_space": "irreps",
            "d6_convention": "z_axis",
            "max_rows": None,
        },
    ]

    for job in jobs:
        out_path = _default_output_path(
            group_name=job["group_name"],
            resolution=job["resolution"],
            feature_space=job["feature_space"],
            d6_convention=job["d6_convention"],
        )
        build_local_iso_lookup_table(
            group_name=job["group_name"],
            out_path=out_path,
            resolution=job["resolution"],
            method=job["method"],
            feature_space=job["feature_space"],
            d6_convention=job["d6_convention"],
            device=device,
            dtype=dtype,
            chunk_size=32768,
            max_rows=job["max_rows"],
            store_quats_convention="passive",
        )


if __name__ == "__main__":
    main()
