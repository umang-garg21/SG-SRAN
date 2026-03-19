"""
Run stage-wise group-property diagnostics on real LR quaternion patches for both HCP and FCC.

This script reuses the same layer-by-layer checks as debug_layer_group_properties.py:
1) left SO(3) equivariance
2) right symmetry invariance
3) left symmetry equivariance
4) right symmetry equivariance

No argparse. Edit CONFIG directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPT_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from models.SR_double_conv_SRattn_a1 import IsoEmbeddingSRAttn
from debug_layer_group_properties import (  # type: ignore
    _best_feature_equivariance,
    _err_metrics,
    _left_action,
    _new_stats,
    _print_property_table,
    _right_action,
    _run_stages,
    _sample_random_so3_quat,
    _summarize_stats,
    _update_stats,
)


CONFIG = {
    "device": "cpu",
    "split": "Test",
    "samples_per_dataset": 1,
    "sample_offset": 0,
    "tol_rel": 5e-3,
    "tol_rms": 2e-4,
    "num_so3_trials": 1,
    "num_sym_trials": 2,
    "lr_crop_hw": [16, 16],
    "seed": 0,
    "upsample_factor": 4,
    "use_lr_conv1": True,
    "use_lr_conv2": True,
    "use_attention": True,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "decoder_backend": "optimizing",
    "decoder_cubochoric_resolution": 1,
    "decoder_num_starts": 1,
    "decoder_steps": 0,
    "decoder_lr": 0.05,
    "decoder_method": "cubochoric",
    # Use full decoder lookup table for highest-fidelity diagnostics.
    "decoder_max_table_rows": None,
    "datasets": [
        {
            "name": "hcp_ti64",
            "crystal": "hcp",
            "d6_convention": "z_axis",
            "dataset_root": "/data/warren/materials/materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4",
        },
        {
            "name": "fcc_in718",
            "crystal": "fcc",
            "d6_convention": "z_axis",
            "dataset_root": "/data/warren/materials/materials_data_mount/datasets/IN718_QSR_x4",
        },
    ],
    "out_json": "scripts/diagnostics/layer_group_properties_real_data_both_a1.json",
}

_NAME_RE = re.compile(
    r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
    re.IGNORECASE,
)


def _pair_key(path: Path) -> tuple[str, int] | None:
    m = _NAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("axis").lower(), int(m.group("id"))


def _load_lr_pairs(dataset_root: Path, split: str) -> list[tuple[tuple[str, int], Path, Path]]:
    split_dir = dataset_root / str(split)
    lr_dir = split_dir / "LR_Data"
    hr_dir = split_dir / "HR_Data"
    if not lr_dir.exists():
        raise FileNotFoundError(f"Missing LR directory: {lr_dir}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"Missing HR directory: {hr_dir}")

    lr_map = {}
    for fp in sorted(lr_dir.glob("*.npy")):
        k = _pair_key(fp)
        if k is not None:
            lr_map[k] = fp
    hr_map = {}
    for fp in sorted(hr_dir.glob("*.npy")):
        k = _pair_key(fp)
        if k is not None:
            hr_map[k] = fp

    common = sorted(set(lr_map.keys()).intersection(hr_map.keys()))
    return [(k, lr_map[k], hr_map[k]) for k in common]


def _ensure_hwc_quat(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        out = arr
    elif arr.shape[0] == 4:
        out = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Could not locate quaternion axis of length 4 in {arr.shape}")
    return out.astype(np.float32, copy=False)


def _maybe_cap_decoder_table(model: IsoEmbeddingSRAttn, max_rows: int | None) -> None:
    if max_rows is None:
        return
    decoder = model.decoder
    if not hasattr(decoder, "table_feat"):
        return
    max_rows_i = int(max_rows)
    if max_rows_i <= 0:
        return

    table_feat = getattr(decoder, "table_feat")
    if int(table_feat.shape[0]) <= max_rows_i:
        return

    with torch.no_grad():
        decoder.table_quats = decoder.table_quats[:max_rows_i]
        decoder.table_feat = decoder.table_feat[:max_rows_i]
        decoder.table_feat_norm = decoder.table_feat_norm[:max_rows_i]


def _init_properties(stage_names: list[str]):
    return {
        "left_so3_equivariant": {nm: _new_stats() for nm in stage_names},
        "right_sym_invariant": {nm: _new_stats() for nm in stage_names},
        "left_sym_equivariant": {nm: _new_stats() for nm in stage_names},
        "right_sym_equivariant": {nm: _new_stats() for nm in stage_names},
    }


def _evaluate_dataset(dataset_cfg: dict, cfg: dict) -> dict:
    device = torch.device(str(cfg["device"]))
    dataset_root = Path(str(dataset_cfg["dataset_root"]))
    split = str(cfg["split"])
    pairs = _load_lr_pairs(dataset_root=dataset_root, split=split)
    if len(pairs) == 0:
        raise RuntimeError(f"No LR/HR pairs found at {dataset_root}/{split}")

    offset = max(0, int(cfg["sample_offset"]))
    n_take = max(1, int(cfg["samples_per_dataset"]))
    selected = pairs[offset : offset + n_take]
    if len(selected) == 0:
        raise RuntimeError(
            f"Requested samples start at offset={offset}, but only {len(pairs)} pair(s) are available."
        )

    model = IsoEmbeddingSRAttn(
        crystal=str(dataset_cfg["crystal"]),
        d6_convention=str(dataset_cfg.get("d6_convention", "z_axis")),
        device=device,
        upsample_factor=int(cfg["upsample_factor"]),
        use_lr_conv1=bool(cfg.get("use_lr_conv1", True)),
        use_lr_conv2=bool(cfg.get("use_lr_conv2", True)),
        use_attention=bool(cfg.get("use_attention", True)),
        num_hr_attn_blocks=int(cfg["num_hr_attn_blocks"]),
        hr_attn_num_channels=int(cfg["hr_attn_num_channels"]),
        hr_attn_block_size=int(cfg["hr_attn_block_size"]),
        decoder_cubochoric_resolution=int(cfg["decoder_cubochoric_resolution"]),
        decoder_num_starts=int(cfg["decoder_num_starts"]),
        decoder_steps=int(cfg["decoder_steps"]),
        decoder_lr=float(cfg["decoder_lr"]),
        decoder_method=str(cfg["decoder_method"]),
        decoder_backend=str(cfg["decoder_backend"]),
    ).eval()
    _maybe_cap_decoder_table(model, cfg.get("decoder_max_table_rows"))

    stage_names: list[str] | None = None
    properties = None
    sample_records = []

    seed_base = int(cfg["seed"]) * 10_000
    tol_rel = float(cfg["tol_rel"])
    tol_rms = float(cfg["tol_rms"])

    sym_ops = model.encoder.sym_ops.detach()
    sym_candidates = sym_ops[1:] if sym_ops.shape[0] > 1 else sym_ops
    n_sym = min(int(cfg["num_sym_trials"]), int(sym_candidates.shape[0]))
    sym_list = [sym_candidates[i] for i in range(n_sym)]
    if len(sym_list) == 0:
        raise RuntimeError("No symmetry operators available for diagnostics.")

    for sample_i, (pair_key, lr_fp, hr_fp) in enumerate(selected):
        lr_arr = _ensure_hwc_quat(np.load(lr_fp))
        crop_hw = cfg.get("lr_crop_hw", None)
        if crop_hw is not None:
            ch, cw = int(crop_hw[0]), int(crop_hw[1])
            lr_arr = lr_arr[:ch, :cw, :]
        h, w, _ = lr_arr.shape
        q_base = torch.from_numpy(lr_arr.reshape(-1, 4)).to(device=device, dtype=torch.float32)
        lr_shape = (h, w)

        base_stages = _run_stages(model, q_base, lr_shape)
        if stage_names is None:
            stage_names = [s["name"] for s in base_stages]
            properties = _init_properties(stage_names)
        assert stage_names is not None
        assert properties is not None
        base_map = {s["name"]: s for s in base_stages}

        # 1) Left SO(3) equivariance.
        for t in range(int(cfg["num_so3_trials"])):
            g = _sample_random_so3_quat(
                seed=seed_base + sample_i * 100 + t,
                device=device,
            )
            trans_map = {
                s["name"]: s for s in _run_stages(model, _left_action(q_base, g), lr_shape)
            }
            for nm in stage_names:
                s_ref = base_map[nm]
                s_tr = trans_map[nm]
                if s_ref["kind"] == "feature":
                    rel, rms, variant = _best_feature_equivariance(
                        s_ref["tensor"],
                        s_tr["tensor"],
                        s_ref["irreps"],
                        g,
                    )
                else:
                    rel, rms = _err_metrics(s_tr["tensor"], _left_action(s_ref["tensor"], g))
                    variant = "quat_left_mult"
                passed = (rel <= tol_rel) or (rms <= tol_rms)
                _update_stats(properties["left_so3_equivariant"][nm], rel, rms, passed, variant)

        # 2) Right symmetry invariance.
        for s in sym_list:
            trans_map = {
                st["name"]: st for st in _run_stages(model, _right_action(q_base, s), lr_shape)
            }
            for nm in stage_names:
                s_ref = base_map[nm]
                s_tr = trans_map[nm]
                rel, rms = _err_metrics(s_tr["tensor"], s_ref["tensor"])
                passed = (rel <= tol_rel) or (rms <= tol_rms)
                _update_stats(properties["right_sym_invariant"][nm], rel, rms, passed, "identity")

        # 3) Left symmetry equivariance.
        for s in sym_list:
            trans_map = {
                st["name"]: st for st in _run_stages(model, _left_action(q_base, s), lr_shape)
            }
            for nm in stage_names:
                s_ref = base_map[nm]
                s_tr = trans_map[nm]
                if s_ref["kind"] == "feature":
                    rel, rms, variant = _best_feature_equivariance(
                        s_ref["tensor"],
                        s_tr["tensor"],
                        s_ref["irreps"],
                        s,
                    )
                else:
                    rel, rms = _err_metrics(s_tr["tensor"], _left_action(s_ref["tensor"], s))
                    variant = "quat_left_mult"
                passed = (rel <= tol_rel) or (rms <= tol_rms)
                _update_stats(properties["left_sym_equivariant"][nm], rel, rms, passed, variant)

        # 4) Right symmetry equivariance.
        for s in sym_list:
            trans_map = {
                st["name"]: st for st in _run_stages(model, _right_action(q_base, s), lr_shape)
            }
            for nm in stage_names:
                s_ref = base_map[nm]
                s_tr = trans_map[nm]
                if s_ref["kind"] == "feature":
                    rel, rms, variant = _best_feature_equivariance(
                        s_ref["tensor"],
                        s_tr["tensor"],
                        s_ref["irreps"],
                        s,
                    )
                else:
                    rel, rms = _err_metrics(s_tr["tensor"], _right_action(s_ref["tensor"], s))
                    variant = "quat_right_mult"
                passed = (rel <= tol_rel) or (rms <= tol_rms)
                _update_stats(properties["right_sym_equivariant"][nm], rel, rms, passed, variant)

        sample_records.append(
            {
                "pair_key": {"axis": pair_key[0], "id": int(pair_key[1])},
                "lr_file": str(lr_fp),
                "hr_file": str(hr_fp),
                "lr_shape_hwc": [int(h), int(w), 4],
            }
        )

    assert stage_names is not None
    assert properties is not None
    results = {
        prop: {nm: _summarize_stats(st) for nm, st in by_stage.items()}
        for prop, by_stage in properties.items()
    }

    return {
        "name": str(dataset_cfg["name"]),
        "crystal": str(dataset_cfg["crystal"]),
        "dataset_root": str(dataset_root),
        "split": split,
        "stage_names": stage_names,
        "num_available_pairs": len(pairs),
        "num_selected_pairs": len(selected),
        "samples": sample_records,
        "results": results,
    }


def main() -> None:
    dataset_reports = []
    for ds_cfg in CONFIG["datasets"]:
        report = _evaluate_dataset(ds_cfg, CONFIG)
        dataset_reports.append(report)

        print("\n" + "=" * 90)
        print(
            f"Dataset: {report['name']} | crystal={report['crystal']} | split={report['split']} "
            f"| samples={report['num_selected_pairs']}/{report['num_available_pairs']}"
        )
        for prop_name, by_stage in report["results"].items():
            _print_property_table(prop_name, by_stage)

    out = {
        "config": CONFIG,
        "dataset_reports": dataset_reports,
    }
    out_path = Path(str(CONFIG["out_json"])).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
