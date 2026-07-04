#!/usr/bin/env python3
"""Prepare zero-shot learned-baseline inference configs.

The source checkpoints stay in their original IN718 / Ti-6Al-4V experiment
directories.  These generated configs only replace the dataset root with an
out-of-distribution target so the standard inference scripts can be reused.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONI_ROOT = Path("/data/home/umang/Materials/Materials_data_mount/datasets/Scan1_x250_QSR_x4")
TI7_ROOT = Path("/data/home/umang/Materials/Materials_data_mount/datasets/Ti7_deformed_4x4")

JANGID_METHODS = {
    "QEDSR": ("IN718/qedsr_4x4_01", "inference/infer_jangid_baseline.py"),
    "Q-RBSA-adapted": ("IN718/qrbsa_4x4_300ep_01", "inference/infer_jangid_baseline.py"),
    "RCAN": ("IN718/rcan_4x4_300ep_01", "inference/infer_jangid_baseline.py"),
    "SAN": ("IN718/san_4x4_300ep_01", "inference/infer_jangid_baseline.py"),
    "HAN": ("IN718/han_4x4_300ep_01", "inference/infer_jangid_baseline.py"),
}
ATINDAMA = {
    "Atindama inpainting": ("IN718/atindama_inpainting_4x4_01", "inference/infer_atindama_inpainting.py")
}


def write_config(source_rel: str, target_root: Path, config_name: str) -> Path:
    exp_dir = ROOT / "experiments" / source_rel
    cfg_path = exp_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["dataset_root"] = str(target_root)
    cfg["preload"] = False
    cfg["preload_torch"] = False
    cfg["num_workers"] = 0
    cfg["inference_num_workers"] = 0
    cfg["persistent_workers"] = False
    cfg["eval_batch_size"] = min(int(cfg.get("eval_batch_size", 1)), 1)
    cfg["zeroshot_source_config"] = str(cfg_path.relative_to(ROOT))
    cfg["zeroshot_target_dataset_root"] = str(target_root)
    out = exp_dir / config_name
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    return out


def main() -> None:
    manifest: list[dict[str, object]] = []
    target = {
        "name": "CoNi_Scan1_x250_4x4",
        "dataset_root": CONI_ROOT,
        "split": "Train",
        "out_root": ROOT / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4",
        "config_name": "config_zeroshot_coni_4x4.json",
    }
    if not (target["dataset_root"] / "dataset_info.json").exists():
        raise FileNotFoundError(target["dataset_root"])

    methods = {**ATINDAMA, **JANGID_METHODS}
    for method, (source_rel, script) in methods.items():
        config_path = write_config(source_rel, target["dataset_root"], target["config_name"])
        exp_dir = ROOT / "experiments" / source_rel
        out_dir = target["out_root"] / method.lower().replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        checkpoint = exp_dir / "checkpoints/best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        manifest.append(
            {
                "target": target["name"],
                "method": method,
                "source_experiment": str(exp_dir.relative_to(ROOT)),
                "script": script,
                "config": str(config_path.relative_to(exp_dir)),
                "checkpoint": "best_model.pt",
                "split": target["split"],
                "out_dir": str(out_dir.relative_to(ROOT)),
            }
        )

    manifest_path = ROOT / "analysis/out/zeroshot_learned_baselines_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"runs": manifest, "ti7_dataset_present": TI7_ROOT.exists()}, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    for idx, row in enumerate(manifest):
        print(
            idx,
            row["method"],
            "python",
            row["script"],
            "--exp_dir",
            row["source_experiment"],
            "--config",
            row["config"],
            "--checkpoint best_model.pt --split",
            row["split"],
            "--out_dir",
            row["out_dir"],
        )
    if not TI7_ROOT.exists():
        print(f"Ti7 target root not present: {TI7_ROOT}")


if __name__ == "__main__":
    main()
