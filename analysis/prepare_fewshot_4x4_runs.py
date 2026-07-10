#!/usr/bin/env python3
"""Prepare 10/90 few-shot adaptation runs for the 4x4 OOD datasets.

The generated dataset roots use symlinks into the original target datasets:
Train and Val contain the selected adaptation subset, while Test contains the
disjoint held-out subset.  Each model is initialized from its source-domain
best checkpoint and fine-tuned with a fresh optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")

PAIR_RE = re.compile(
    r"^(?P<ds>.+)_(?P<src_split>train|val|test)_(?P<which>hr|lr|org)_"
    r"(?P<axis>[xyz])_block_(?P<block>\d+)\.npy$",
    re.IGNORECASE,
)


TARGETS = OrderedDict(
    [
        (
            "coni_x250",
            {
                "name": "CoNi_x250",
                "task": "IN718 -> CoNi x250",
                "dataset_root": Path(
                    "/data/home/umang/Materials/Materials_data_mount/datasets/Scan1_x250_QSR_x4"
                ),
                "out_root": ROOT / "experiments/Few_shot_performance_CoNi_x250",
                "source_family": "IN718",
                "symmetry": "Oh",
                "crystal": "fcc",
                "ocrp_source_rel": (
                    "IN718/direct_reynolds_isometric_seed_runs/"
                    "ocrp_direct_reynolds_isometric_l4_s42"
                ),
                "ocrp_slug": "ocrp_direct_reynolds_isometric_l4",
            },
        ),
        (
            "ti7_deformed",
            {
                "name": "Ti7_deformed",
                "task": "Ti-6Al-4V -> Ti7 deformed",
                "dataset_root": Path(
                    "/data/home/umang/Materials/Materials_data_mount/datasets/Ti7_deformed_4x4"
                ),
                "out_root": ROOT / "experiments/Few_shot_performance_Ti7_deformed",
                "source_family": "Ti_Al_1pct",
                "symmetry": "D6h",
                "crystal": "hcp",
                "ocrp_source_rel": (
                    "Ti_Al_1pct/direct_reynolds_isometric_seed_runs/"
                    "ocrp_direct_reynolds_isometric_l6_s42"
                ),
                "ocrp_slug": "ocrp_direct_reynolds_isometric_l6",
            },
        ),
        (
            "ti64_dic_mclean",
            {
                "name": "Ti64",
                "task": "Ti-6Al-4V -> Ti64",
                "dataset_root": Path(
                    "/data/home/umang/Materials/Materials_data_mount/datasets/"
                    "Ti64_DIC_Mclean_QSR_x4"
                ),
                "out_root": ROOT / "experiments/Few_shot_performance_Ti64_DIC_Mclean",
                "source_family": "Ti_Al_1pct",
                "symmetry": "D6h",
                "crystal": "hcp",
                "ocrp_source_rel": (
                    "Ti_Al_1pct/direct_reynolds_isometric_seed_runs/"
                    "ocrp_direct_reynolds_isometric_l6_s42"
                ),
                "ocrp_slug": "ocrp_direct_reynolds_isometric_l6",
            },
        ),
    ]
)


BASELINE_SOURCES = {
    "IN718": OrderedDict(
        [
            ("Atindama inpainting", ("atindama", "atindama_inpainting_4x4_01")),
            ("EDSR", ("jangid", "edsr_4x4_01")),
            ("QEDSR", ("jangid", "qedsr_4x4_01")),
            ("Q-RBSA-adapted", ("jangid", "qrbsa_4x4_300ep_01")),
            ("RCAN", ("jangid", "rcan_4x4_300ep_01")),
            ("SAN", ("jangid", "san_4x4_300ep_01")),
            ("HAN", ("jangid", "han_4x4_300ep_01")),
        ]
    ),
    "Ti_Al_1pct": OrderedDict(
        [
            ("Atindama inpainting", ("atindama", "atindama_inpainting_4x4_01")),
            ("EDSR", ("jangid", "edsr_4x4_01")),
            ("QEDSR", ("jangid", "qedsr_4x4_01")),
            ("Q-RBSA-adapted", ("jangid", "qrbsa_adapted_4x4_300ep_01")),
            ("RCAN", ("jangid", "rcan_4x4_300ep_01")),
            ("SAN", ("jangid", "san_4x4_300ep_01")),
            ("HAN", ("jangid", "han_4x4_300ep_01")),
        ]
    ),
}


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )


def _sample_key(path: Path) -> tuple[str, str, str, int] | None:
    match = PAIR_RE.match(path.name)
    if not match:
        return None
    return (
        match.group("ds"),
        match.group("src_split").lower(),
        match.group("axis").lower(),
        int(match.group("block")),
    )


def _collect_pairs(dataset_root: Path) -> list[dict[str, object]]:
    lr_by_key: dict[tuple[str, str, str, int], Path] = {}
    hr_by_key: dict[tuple[str, str, str, int], Path] = {}
    org_by_key: dict[tuple[str, str, str, int], Path] = {}

    for split in ("Train", "Val", "Test"):
        for path in sorted((dataset_root / split / "LR_Data").glob("*.npy")):
            key = _sample_key(path)
            if key is not None:
                lr_by_key[key] = path
        for path in sorted((dataset_root / split / "HR_Data").glob("*.npy")):
            key = _sample_key(path)
            if key is not None:
                hr_by_key[key] = path
        for path in sorted((dataset_root / split / "Original_Data").glob("*.npy")):
            key = _sample_key(path)
            if key is not None:
                org_by_key[key] = path

    keys = sorted(lr_by_key.keys() & hr_by_key.keys())
    if not keys:
        raise RuntimeError(f"No LR/HR pairs found in {dataset_root}")
    return [
        {
            "key": key,
            "lr": lr_by_key[key],
            "hr": hr_by_key[key],
            "org": org_by_key.get(key),
        }
        for key in keys
    ]


def _link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def _write_split_files(
    dataset_root: Path,
    split: str,
    pairs: list[dict[str, object]],
) -> None:
    for pair in pairs:
        _link_file(Path(pair["lr"]), dataset_root / split / "LR_Data" / Path(pair["lr"]).name)
        _link_file(Path(pair["hr"]), dataset_root / split / "HR_Data" / Path(pair["hr"]).name)
        org = pair.get("org")
        if org is not None:
            _link_file(Path(org), dataset_root / split / "Original_Data" / Path(org).name)


def _split_globs(dataset_root: Path, split: str) -> dict[str, str]:
    return {
        "ORG_glob": str(dataset_root / split / "Original_Data" / "*.npy"),
        "HR_glob": str(dataset_root / split / "HR_Data" / "*.npy"),
        "LR_glob": str(dataset_root / split / "LR_Data" / "*.npy"),
    }


def _key_to_string(key: tuple[str, str, str, int]) -> str:
    dataset, src_split, axis, block = key
    return f"{dataset}:{src_split}:{axis}:{block}"


def _prepare_fewshot_dataset(
    target_key: str,
    target: dict[str, object],
    *,
    fraction: float,
    seed: int,
) -> tuple[Path, dict[str, object]]:
    source_root = Path(target["dataset_root"])
    all_pairs = _collect_pairs(source_root)
    rng = random.Random(seed)
    shuffled = list(all_pairs)
    rng.shuffle(shuffled)
    train_count = max(1, int(math.ceil(len(shuffled) * fraction)))
    if train_count >= len(shuffled):
        raise ValueError(
            f"Few-shot split for {target_key} would leave no held-out samples: "
            f"n={len(shuffled)}, fraction={fraction}"
        )
    train_pairs = sorted(shuffled[:train_count], key=lambda row: row["key"])
    test_pairs = sorted(shuffled[train_count:], key=lambda row: row["key"])

    tag = f"fewshot{int(round(fraction * 100)):02d}_s{seed}"
    dataset_root = (
        ROOT
        / "experiments/Few_shot_adaptation_4x4/datasets"
        / f"{target_key}_{tag}"
    )
    for split in ("Train", "Val", "Test"):
        for sub in ("LR_Data", "HR_Data", "Original_Data"):
            (dataset_root / split / sub).mkdir(parents=True, exist_ok=True)
            for old in (dataset_root / split / sub).glob("*.npy"):
                old.unlink()

    _write_split_files(dataset_root, "Train", train_pairs)
    _write_split_files(dataset_root, "Val", train_pairs)
    _write_split_files(dataset_root, "Test", test_pairs)

    train_org = sum(1 for row in train_pairs if row.get("org") is not None)
    test_org = sum(1 for row in test_pairs if row.get("org") is not None)
    info: dict[str, object] = {
        "name": f"{target['name']}_{tag}",
        "source_dataset_root": str(source_root),
        "symmetry": str(target["symmetry"]),
        "scale": [4, 4],
        "fewshot": {
            "fraction_requested": float(fraction),
            "seed": int(seed),
            "selection_policy": "shuffle all LR/HR pairs with seed, use ceil(fraction*n) for adaptation",
            "validation_policy": "Val duplicates the adaptation subset; Test is disjoint held-out data.",
            "num_total_pairs": int(len(shuffled)),
            "num_adaptation_pairs": int(len(train_pairs)),
            "num_heldout_pairs": int(len(test_pairs)),
            "actual_adaptation_fraction": float(len(train_pairs) / len(shuffled)),
            "adaptation_keys": [_key_to_string(row["key"]) for row in train_pairs],
            "heldout_keys": [_key_to_string(row["key"]) for row in test_pairs],
        },
        "counts": {
            "Train": {"hr": len(train_pairs), "lr": len(train_pairs), "org": train_org},
            "Val": {"hr": len(train_pairs), "lr": len(train_pairs), "org": train_org},
            "Test": {"hr": len(test_pairs), "lr": len(test_pairs), "org": test_org},
        },
        "splits": {
            split: _split_globs(dataset_root, split)
            for split in ("Train", "Val", "Test")
        },
    }
    (dataset_root / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return dataset_root, info


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _common_finetune_config(
    cfg: dict,
    *,
    target: dict[str, object],
    dataset_root: Path,
    source_config: Path,
    source_checkpoint: Path,
    epochs: int,
    lr_scale: float,
    train_count: int,
) -> dict:
    cfg = dict(cfg)
    cfg["dataset_root"] = str(dataset_root)
    cfg["symmetry"] = str(target["symmetry"])
    cfg["symmetry_group"] = str(target["symmetry"])
    cfg["crystal"] = str(target["crystal"])
    cfg["scale"] = [4, 4]
    cfg["upsample_factor"] = [4, 4]
    cfg["epochs"] = int(epochs)
    cfg["lr"] = float(cfg.get("lr", 3e-4)) * float(lr_scale)
    cfg["batch_size"] = max(1, min(int(cfg.get("batch_size", 2)), int(train_count)))
    cfg["eval_batch_size"] = 1
    cfg["preload"] = False
    cfg["preload_torch"] = False
    cfg["num_workers"] = 0
    cfg["inference_num_workers"] = 0
    cfg["persistent_workers"] = False
    cfg["prefetch_factor"] = 2
    cfg["pin_memory"] = True
    cfg["save_every"] = 0
    cfg["viz_every"] = 0
    cfg["save_last_checkpoint"] = False
    cfg["save_epoch_checkpoints"] = False
    cfg["final_viz"] = False
    cfg["plot_loss_curves"] = False
    cfg["fewshot_source_config"] = str(source_config.relative_to(ROOT))
    cfg["fewshot_source_checkpoint"] = str(source_checkpoint.relative_to(ROOT))
    cfg["fewshot_target_dataset_root"] = str(dataset_root)
    cfg["fewshot_target_name"] = str(target["name"])
    cfg["fewshot_task"] = str(target["task"])
    cfg["fewshot_val_policy"] = "Val duplicates Train adaptation subset; Test is disjoint."
    for key in ("take_first", "train_take_first", "val_take_first", "test_take_first"):
        cfg.pop(key, None)
    if isinstance(cfg.get("logging"), dict):
        cfg["logging"] = dict(cfg["logging"])
        cfg["logging"]["tensorboard"] = False
        cfg["logging"]["save_best_only"] = True
        cfg["logging"]["val_freq"] = 1
    return cfg


def _command_for_train(
    *,
    kind: str,
    exp_dir: Path,
    config_name: str,
    source_checkpoint: Path,
    gpu: str,
) -> list[str]:
    if kind == "ocrp":
        return [
            str(PYTHON),
            str(ROOT / "training/train_iso_embedding_ocrp.py"),
            "--exp_dir",
            str(exp_dir),
            "--config",
            config_name,
            "--init_checkpoint",
            str(source_checkpoint),
            "--gpu_ids",
            gpu,
        ]
    if kind == "atindama":
        return [
            str(PYTHON),
            str(ROOT / "training/train_atindama_inpainting.py"),
            "--exp_dir",
            str(exp_dir),
            "--config",
            config_name,
            "--init_checkpoint",
            str(source_checkpoint),
            "--gpu",
            gpu,
        ]
    return [
        str(PYTHON),
        str(ROOT / "training/train_jangid_baseline.py"),
        "--exp_dir",
        str(exp_dir),
        "--config",
        config_name,
        "--init_checkpoint",
        str(source_checkpoint),
        "--skip_viz",
        "--gpu",
        gpu,
    ]


def _command_for_inference(
    *,
    kind: str,
    exp_dir: Path,
    config_name: str,
    gpu: str,
) -> list[str]:
    out_dir = exp_dir / "inference/test_best"
    if kind == "ocrp":
        return [
            str(PYTHON),
            str(ROOT / "inference/infer_iso_embedding_sr_attn.py"),
            "--exp_dir",
            str(exp_dir),
            "--config",
            config_name,
            "--checkpoint",
            "best_model.pt",
            "--split",
            "Test",
            "--out_dir",
            str(out_dir),
            "--skip_ipf",
            "--gpu_ids",
            gpu,
        ]
    script = (
        "inference/infer_atindama_inpainting.py"
        if kind == "atindama"
        else "inference/infer_jangid_baseline.py"
    )
    return [
        str(PYTHON),
        str(ROOT / script),
        "--exp_dir",
        str(exp_dir),
        "--config",
        config_name,
        "--checkpoint",
        "best_model.pt",
        "--split",
        "Test",
        "--out_dir",
        str(out_dir),
        "--max_visualizations",
        "0",
        "--gpu",
        gpu,
    ]


def _write_method_config(
    *,
    method_name: str,
    kind: str,
    source_exp: Path,
    source_config_name: str,
    exp_dir: Path,
    target: dict[str, object],
    dataset_root: Path,
    epochs: int,
    lr_scale: float,
    train_count: int,
) -> tuple[Path, Path]:
    source_config = source_exp / source_config_name
    source_checkpoint = source_exp / "checkpoints/best_model.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    cfg = _common_finetune_config(
        _load_config(source_config),
        target=target,
        dataset_root=dataset_root,
        source_config=source_config,
        source_checkpoint=source_checkpoint,
        epochs=epochs,
        lr_scale=lr_scale,
        train_count=train_count,
    )
    cfg["fewshot_method"] = method_name
    cfg["checkpoints_dir"] = str(exp_dir / "checkpoints")
    if kind == "ocrp":
        cfg.setdefault("model", {})
        cfg["logging"] = {
            "tensorboard": False,
            "save_best_only": True,
            "val_freq": 1,
        }

    exp_dir.mkdir(parents=True, exist_ok=True)
    config_name = "config_new.json" if kind == "ocrp" else "config.json"
    config_path = exp_dir / config_name
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return config_path, source_checkpoint


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=[*TARGETS.keys(), "all"],
        default="all",
        help="Target dataset to prepare.",
    )
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--gpu", default="0", help="GPU id written into generated commands.")
    args = parser.parse_args()

    selected_targets = (
        TARGETS
        if args.target == "all"
        else OrderedDict([(args.target, TARGETS[args.target])])
    )

    runs: list[dict[str, object]] = []
    train_lines: list[str] = []
    infer_lines: list[str] = []
    full_lines: list[str] = []

    for target_key, target in selected_targets.items():
        dataset_root, dataset_info = _prepare_fewshot_dataset(
            target_key,
            target,
            fraction=float(args.fraction),
            seed=int(args.seed),
        )
        tag = f"fewshot{int(round(float(args.fraction) * 100)):02d}_s{int(args.seed)}"
        train_count = int(dataset_info["counts"]["Train"]["hr"])  # type: ignore[index]
        test_count = int(dataset_info["counts"]["Test"]["hr"])  # type: ignore[index]
        target_out_root = Path(target["out_root"]) / tag

        family = str(target["source_family"])
        for method_name, (kind, source_leaf) in BASELINE_SOURCES[family].items():
            source_exp = ROOT / "experiments" / family / source_leaf
            exp_dir = target_out_root / "learned_baselines_4x4" / _slug(method_name)
            config_path, source_checkpoint = _write_method_config(
                method_name=method_name,
                kind=kind,
                source_exp=source_exp,
                source_config_name="config.json",
                exp_dir=exp_dir,
                target=target,
                dataset_root=dataset_root,
                epochs=int(args.epochs),
                lr_scale=float(args.lr_scale),
                train_count=train_count,
            )
            train_cmd = _command_for_train(
                kind=kind,
                exp_dir=exp_dir,
                config_name=config_path.name,
                source_checkpoint=source_checkpoint,
                gpu=str(args.gpu),
            )
            infer_cmd = _command_for_inference(
                kind=kind,
                exp_dir=exp_dir,
                config_name=config_path.name,
                gpu=str(args.gpu),
            )
            train_lines.append(_shell_join(train_cmd))
            infer_lines.append(_shell_join(infer_cmd))
            full_lines.append(_shell_join(train_cmd) + "\n" + _shell_join(infer_cmd))
            runs.append(
                {
                    "target_key": target_key,
                    "target_name": target["name"],
                    "task": target["task"],
                    "method": method_name,
                    "kind": kind,
                    "experiment": str(exp_dir.relative_to(ROOT)),
                    "config": config_path.name,
                    "source_experiment": str(source_exp.relative_to(ROOT)),
                    "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
                    "fewshot_dataset_root": str(dataset_root.relative_to(ROOT)),
                    "adaptation_samples": train_count,
                    "heldout_samples": test_count,
                    "train_command": train_cmd,
                    "inference_command": infer_cmd,
                }
            )

        ocrp_source_exp = ROOT / "experiments" / str(target["ocrp_source_rel"])
        ocrp_exp_dir = target_out_root / str(target["ocrp_slug"])
        ocrp_config_path, ocrp_source_checkpoint = _write_method_config(
            method_name="OCRP",
            kind="ocrp",
            source_exp=ocrp_source_exp,
            source_config_name="config_new.json",
            exp_dir=ocrp_exp_dir,
            target=target,
            dataset_root=dataset_root,
            epochs=int(args.epochs),
            lr_scale=float(args.lr_scale),
            train_count=train_count,
        )
        train_cmd = _command_for_train(
            kind="ocrp",
            exp_dir=ocrp_exp_dir,
            config_name=ocrp_config_path.name,
            source_checkpoint=ocrp_source_checkpoint,
            gpu=str(args.gpu),
        )
        infer_cmd = _command_for_inference(
            kind="ocrp",
            exp_dir=ocrp_exp_dir,
            config_name=ocrp_config_path.name,
            gpu=str(args.gpu),
        )
        train_lines.append(_shell_join(train_cmd))
        infer_lines.append(_shell_join(infer_cmd))
        full_lines.append(_shell_join(train_cmd) + "\n" + _shell_join(infer_cmd))
        runs.append(
            {
                "target_key": target_key,
                "target_name": target["name"],
                "task": target["task"],
                "method": "OCRP",
                "kind": "ocrp",
                "experiment": str(ocrp_exp_dir.relative_to(ROOT)),
                "config": ocrp_config_path.name,
                "source_experiment": str(ocrp_source_exp.relative_to(ROOT)),
                "source_checkpoint": str(ocrp_source_checkpoint.relative_to(ROOT)),
                "fewshot_dataset_root": str(dataset_root.relative_to(ROOT)),
                "adaptation_samples": train_count,
                "heldout_samples": test_count,
                "train_command": train_cmd,
                "inference_command": infer_cmd,
            }
        )

    out_dir = ROOT / "analysis/out"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "fewshot_4x4_manifest.json"
    train_sh = out_dir / "fewshot_4x4_train_commands.sh"
    infer_sh = out_dir / "fewshot_4x4_infer_commands.sh"
    full_sh = out_dir / "fewshot_4x4_train_then_infer_commands.sh"

    manifest = {
        "fraction": float(args.fraction),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr_scale": float(args.lr_scale),
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    train_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(train_lines) + "\n")
    infer_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(infer_lines) + "\n")
    full_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(full_lines) + "\n")
    for path in (train_sh, infer_sh, full_sh):
        path.chmod(0o755)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {train_sh}")
    print(f"Wrote {infer_sh}")
    print(f"Wrote {full_sh}")
    for idx, run in enumerate(runs):
        print(
            f"{idx:02d} {run['target_name']} {run['method']} "
            f"train={run['adaptation_samples']} test={run['heldout_samples']} "
            f"exp={run['experiment']}"
        )


if __name__ == "__main__":
    main()
