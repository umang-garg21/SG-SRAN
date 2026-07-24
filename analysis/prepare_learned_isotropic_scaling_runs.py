#!/usr/bin/env python3
"""Prepare missing learned-baseline isotropic scaling runs.

This fills the scaling protocol for learned baselines at 2x2 and 8x8. The 4x4
seed-42 learned runs already exist under experiments/<material>/seed_runs and
are reused by the plotting script.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")

MATERIALS = OrderedDict(
    [
        (
            "IN718",
            {
                "dataset_stem": "IN718_QSR",
                "symmetry": "Oh",
                "source_root": ROOT / "experiments/IN718/seed_runs",
            },
        ),
        (
            "Ti_Al_1pct",
            {
                "dataset_stem": "Ti_Al_1pct_QSR",
                "symmetry": "D6h",
                "source_root": ROOT / "experiments/Ti_Al_1pct/seed_runs",
            },
        ),
    ]
)

METHODS = OrderedDict(
    [
        ("Atindama inpainting", {"slug": "atindama", "source": "atindama_4x4_s42", "kind": "atindama"}),
        ("EDSR", {"slug": "edsr", "source": "edsr_4x4_s42", "kind": "jangid"}),
        ("QEDSR", {"slug": "qedsr", "source": "qedsr_4x4_s42", "kind": "jangid"}),
        ("Q-RBSA-adapted", {"slug": "qrbsa", "source": "qrbsa_4x4_s42", "kind": "jangid"}),
        ("RCAN", {"slug": "rcan", "source": "rcan_4x4_s42", "kind": "jangid"}),
        ("SAN", {"slug": "san", "source": "san_4x4_s42", "kind": "jangid"}),
        ("HAN", {"slug": "han", "source": "han_4x4_s42", "kind": "jangid"}),
    ]
)

SCALES = (2, 8)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def status_gate(exp_dir: Path, train_cmd: str, infer_cmd: str) -> str:
    summary_path = exp_dir / "inference/test_best/summary.json"
    best_ckpt = exp_dir / "checkpoints/best_model.pt"
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(exp_dir / 'logs'))}",
            f"if [ -f {shlex.quote(str(summary_path))} ]; then",
            f"  echo IDONE > {shlex.quote(str(exp_dir / 'ISTATUS'))}",
            f"  echo DONE > {shlex.quote(str(exp_dir / 'STATUS'))}",
            "fi",
            f"if [ ! -f {shlex.quote(str(best_ckpt))} ] && [ \"$(cat {shlex.quote(str(exp_dir / 'STATUS'))} 2>/dev/null || true)\" != \"DONE\" ]; then",
            f"  echo RUNNING > {shlex.quote(str(exp_dir / 'STATUS'))}",
            f"  {train_cmd} > {shlex.quote(str(exp_dir / 'logs/train_scaling_seed42.log'))} 2>&1",
            f"  echo TRAIN_DONE > {shlex.quote(str(exp_dir / 'STATUS'))}",
            "fi",
            f"if [ \"$(cat {shlex.quote(str(exp_dir / 'ISTATUS'))} 2>/dev/null || true)\" != \"IDONE\" ]; then",
            f"  echo IRUNNING > {shlex.quote(str(exp_dir / 'ISTATUS'))}",
            f"  {infer_cmd} > {shlex.quote(str(exp_dir / 'logs/infer_test_best_scaling_seed42.log'))} 2>&1",
            f"  echo IDONE > {shlex.quote(str(exp_dir / 'ISTATUS'))}",
            f"  echo DONE > {shlex.quote(str(exp_dir / 'STATUS'))}",
            "fi",
        ]
    )


def command_for_train(kind: str, exp_dir: Path, gpu: str) -> list[str]:
    script = "training/train_atindama_inpainting.py" if kind == "atindama" else "training/train_jangid_baseline.py"
    cmd = [
        str(PYTHON),
        str(ROOT / script),
        "--exp_dir",
        str(exp_dir),
        "--config",
        "config.json",
        "--gpu",
        gpu,
    ]
    if kind != "atindama":
        cmd.append("--skip_viz")
    return cmd


def command_for_inference(kind: str, exp_dir: Path, gpu: str) -> list[str]:
    script = "inference/infer_atindama_inpainting.py" if kind == "atindama" else "inference/infer_jangid_baseline.py"
    return [
        str(PYTHON),
        str(ROOT / script),
        "--exp_dir",
        str(exp_dir),
        "--config",
        "config.json",
        "--checkpoint",
        "best_model.pt",
        "--split",
        "Test",
        "--out_dir",
        str(exp_dir / "inference/test_best"),
        "--max_visualizations",
        "0",
        "--gpu",
        gpu,
    ]


def prepare_run(material: str, material_spec: dict, method_name: str, method_spec: dict, scale: int) -> dict:
    source_exp = Path(material_spec["source_root"]) / str(method_spec["source"])
    source_config = source_exp / "config.json"
    if not source_config.exists():
        raise FileNotFoundError(source_config)

    dataset_root = (
        ROOT
        / "experiments/direct_reynolds_isometric_scaling/datasets"
        / f"{material_spec['dataset_stem']}_x{scale}"
    )
    if not (dataset_root / "dataset_info.json").exists():
        raise FileNotFoundError(dataset_root / "dataset_info.json")

    exp_dir = (
        ROOT
        / "experiments/direct_reynolds_isometric_scaling"
        / material
        / f"{method_spec['slug']}_x{scale}_s42"
    )

    cfg = load_json(source_config)
    cfg["dataset_root"] = str(dataset_root)
    cfg["scale"] = [scale, scale]
    cfg["seed"] = 42
    cfg["symmetry_group"] = str(material_spec["symmetry"])
    cfg["isotropic_scaling_protocol"] = "learned_baseline_seed42_x2_x8"
    cfg["source_config"] = str(source_config.relative_to(ROOT))
    cfg["method_name"] = method_name
    cfg.pop("expected_trainable_params", None)
    cfg.pop("checkpoints_dir", None)
    if method_spec["kind"] == "atindama":
        cfg["task"] = f"periodic_{scale}x{scale}_inpainting"

    write_json(exp_dir / "config.json", cfg)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "STATUS").write_text(
        ((exp_dir / "STATUS").read_text(encoding="utf-8") if (exp_dir / "STATUS").exists() else "PENDING\n"),
        encoding="utf-8",
    )
    (exp_dir / "ISTATUS").write_text(
        ((exp_dir / "ISTATUS").read_text(encoding="utf-8") if (exp_dir / "ISTATUS").exists() else "IPENDING\n"),
        encoding="utf-8",
    )

    return {
        "material": material,
        "method": method_name,
        "scale": f"{scale}x{scale}",
        "exp_dir": str(exp_dir.relative_to(ROOT)),
        "kind": method_spec["kind"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,6")
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")

    launch_dir = ROOT / "experiments/direct_reynolds_isometric_scaling/_launch"
    launch_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    job_blocks: list[str] = []
    for material, material_spec in MATERIALS.items():
        for scale in SCALES:
            for method_name, method_spec in METHODS.items():
                run = prepare_run(material, material_spec, method_name, method_spec, scale)
                exp_dir = ROOT / run["exp_dir"]
                gpu = gpus[len(job_blocks) % len(gpus)]
                train_cmd = shell_join(command_for_train(run["kind"], exp_dir, gpu))
                infer_cmd = shell_join(command_for_inference(run["kind"], exp_dir, gpu))
                job_blocks.append(
                    "\n".join(
                        [
                            f"echo '[job] {material} {run['scale']} {method_name} on gpu {gpu}'",
                            status_gate(exp_dir, train_cmd, infer_cmd),
                        ]
                    )
                )
                run["gpu"] = gpu
                runs.append(run)

    workers: list[list[str]] = [[] for _ in gpus]
    for idx, block in enumerate(job_blocks):
        workers[idx % len(gpus)].append(block)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "pids=()",
    ]
    for gpu, blocks in zip(gpus, workers, strict=True):
        lines.append("(")
        lines.append("set -euo pipefail")
        lines.append(f"echo '[worker] gpu {gpu} starting {len(blocks)} jobs'")
        lines.extend(blocks)
        lines.append(f"echo '[worker] gpu {gpu} done'")
        lines.append(") &")
        lines.append("pids+=(\"$!\")")
    lines.append("for pid in \"${pids[@]}\"; do wait \"$pid\"; done")
    lines.append("echo '[master] learned scaling jobs complete'")

    launch_path = launch_dir / "run_learned_scaling_seed42_train_infer.sh"
    launch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launch_path.chmod(0o755)
    write_json(launch_dir / "learned_scaling_seed42_runs.json", {"runs": runs})
    print(f"prepared {len(runs)} runs")
    print(f"launch: {launch_path.relative_to(ROOT)}")
    for gpu, blocks in zip(gpus, workers, strict=True):
        print(f"gpu {gpu}: {len(blocks)} jobs")


if __name__ == "__main__":
    main()
