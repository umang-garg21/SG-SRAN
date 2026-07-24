#!/usr/bin/env python3
"""Prepare HCP zero-shot 4x4 inference configs and a runnable manifest.

The source checkpoints remain the Ti-6Al-4V 4x4 checkpoints.  The generated
configs only replace the dataset root and inference safety settings so the
standard inference scripts can be reused without target retraining.
"""
from __future__ import annotations

import argparse
import json
import shlex
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")

TARGETS = OrderedDict(
    [
        (
            "ti7",
            {
                "name": "Ti7_deformed_4x4",
                "task": "Ti-6Al-4V -> Ti7-deformed 4x4",
                "dataset_root": Path(
                    "/data/home/umang/Materials/Materials_data_mount/datasets/Ti7_deformed_4x4"
                ),
                "split": "Train",
                "out_root": ROOT / "experiments/Zero_shot_performance_Ti7_deformed",
                "config_name": "config_zeroshot_ti7_deformed_4x4.json",
            },
        ),
        (
            "ti64_dic_mclean",
            {
                "name": "Ti64",
                "task": "Ti-6Al-4V -> Ti64 4x4",
                "dataset_root": Path(
                    "/data/home/umang/Materials/Materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4"
                ),
                "split": "Test",
                "out_root": ROOT / "experiments/Zero_shot_performance_Ti64_DIC_Mclean",
                "config_name": "config_zeroshot_ti64_dic_mclean_4x4.json",
            },
        ),
    ]
)

METHODS = OrderedDict(
    [
        (
            "Atindama inpainting",
            {
                "source_rel": "Ti_Al_1pct/atindama_inpainting_4x4_01",
                "config": "config.json",
                "script": "inference/infer_atindama_inpainting.py",
                "kind": "atindama",
            },
        ),
        (
            "EDSR",
            {
                "source_rel": "Ti_Al_1pct/edsr_4x4_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "QEDSR",
            {
                "source_rel": "Ti_Al_1pct/qedsr_4x4_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "Q-RBSA-adapted",
            {
                "source_rel": "Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "RCAN",
            {
                "source_rel": "Ti_Al_1pct/rcan_4x4_300ep_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "SAN",
            {
                "source_rel": "Ti_Al_1pct/san_4x4_300ep_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "HAN",
            {
                "source_rel": "Ti_Al_1pct/han_4x4_300ep_01",
                "config": "config.json",
                "script": "inference/infer_jangid_baseline.py",
                "kind": "jangid",
            },
        ),
        (
            "OCRP (ours)",
            {
                "source_rel": "Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42",
                "config": "config_new.json",
                "script": "inference/infer_iso_embedding_sr_attn.py",
                "kind": "ocrp",
            },
        ),
    ]
)


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )


def _load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    return json.loads(info_path.read_text())


def _write_config(method: dict, target: dict) -> Path:
    exp_dir = ROOT / "experiments" / str(method["source_rel"])
    source_config = exp_dir / str(method["config"])
    if not source_config.exists():
        raise FileNotFoundError(source_config)

    cfg = json.loads(source_config.read_text())
    cfg["dataset_root"] = str(target["dataset_root"])
    cfg["symmetry"] = "D6h"
    cfg["symmetry_group"] = "D6h"
    cfg["crystal"] = "hcp"
    cfg["d6_convention"] = cfg.get("d6_convention", "z_axis")
    cfg["scale"] = [4, 4]
    cfg["upsample_factor"] = [4, 4]
    cfg["batch_size"] = 1
    cfg["eval_batch_size"] = 1
    cfg["preload"] = False
    cfg["preload_torch"] = False
    cfg["num_workers"] = 0
    cfg["inference_num_workers"] = 0
    cfg["persistent_workers"] = False
    cfg["prefetch_factor"] = 2
    cfg["pin_memory"] = True
    for key in ("take_first", "train_take_first", "val_take_first", "test_take_first"):
        cfg.pop(key, None)
    cfg["zeroshot_source_config"] = str(source_config.relative_to(ROOT))
    cfg["zeroshot_source_dataset_root"] = json.loads(source_config.read_text()).get("dataset_root")
    cfg["zeroshot_target_dataset_root"] = str(target["dataset_root"])
    cfg["zeroshot_target_name"] = str(target["name"])
    cfg["zeroshot_target_split"] = str(target["split"])
    cfg["zeroshot_task"] = str(target["task"])

    out_path = exp_dir / str(target["config_name"])
    out_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return out_path


def _command_for(method_name: str, method: dict, target: dict, config_path: Path, gpu: str) -> list[str]:
    exp_dir = ROOT / "experiments" / str(method["source_rel"])
    checkpoint = exp_dir / "checkpoints/best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if method["kind"] == "ocrp":
        out_dir = (
            Path(target["out_root"])
            / "ocrp_direct_reynolds_isometric_l6_s42"
            / "inference"
            / f"{str(target['split']).lower()}_best"
        )
        return [
            str(PYTHON),
            str(ROOT / str(method["script"])),
            "--exp_dir",
            str(exp_dir),
            "--config",
            config_path.name,
            "--checkpoint",
            "best_model.pt",
            "--split",
            str(target["split"]),
            "--out_dir",
            str(out_dir),
            "--skip_ipf",
            "--gpu_ids",
            gpu,
        ]

    out_dir = Path(target["out_root"]) / "learned_baselines_4x4" / _slug(method_name)
    cmd = [
        str(PYTHON),
        str(ROOT / str(method["script"])),
        "--exp_dir",
        str(exp_dir),
        "--config",
        config_path.name,
        "--checkpoint",
        "best_model.pt",
        "--split",
        str(target["split"]),
        "--out_dir",
        str(out_dir),
        "--max_visualizations",
        "0",
        "--gpu",
        gpu,
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=[*TARGETS.keys(), "all"],
        default="all",
        help="Which target configs to prepare.",
    )
    parser.add_argument("--gpu", default="0", help="GPU id to put in printed commands.")
    args = parser.parse_args()

    selected = TARGETS if args.target == "all" else OrderedDict([(args.target, TARGETS[args.target])])
    runs: list[dict[str, object]] = []
    shell_lines: list[str] = []
    for target_key, target in selected.items():
        info = _load_dataset_info(Path(target["dataset_root"]))
        split = str(target["split"])
        count = int(info.get("counts", {}).get(split, {}).get("hr", 0))
        if count <= 0:
            raise ValueError(f"{target['name']} has no HR samples for split {split}")

        for method_name, method in METHODS.items():
            config_path = _write_config(method, target)
            cmd = _command_for(method_name, method, target, config_path, str(args.gpu))
            exp_dir = ROOT / "experiments" / str(method["source_rel"])
            runs.append(
                {
                    "target_key": target_key,
                    "target_name": target["name"],
                    "task": target["task"],
                    "split": split,
                    "num_target_samples": count,
                    "method": method_name,
                    "kind": method["kind"],
                    "source_experiment": str(exp_dir.relative_to(ROOT)),
                    "config": str(config_path.relative_to(exp_dir)),
                    "checkpoint": "best_model.pt",
                    "command": cmd,
                }
            )
            shell_lines.append(" ".join(shlex.quote(part) for part in cmd))

    out_json = ROOT / "analysis/out/hcp_zeroshot_4x4_manifest.json"
    out_sh = ROOT / "analysis/out/hcp_zeroshot_4x4_commands.sh"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    out_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(shell_lines) + "\n")
    out_sh.chmod(0o755)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_sh}")
    for idx, run in enumerate(runs):
        print(f"{idx:02d} {run['target_name']} {run['method']} split={run['split']}")


if __name__ == "__main__":
    main()
