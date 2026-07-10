#!/usr/bin/env python3
"""Prepare the Ti64 DIC McLean OCRP w5-local rerun.

This keeps the existing Train/Val/Test split and starts from the completed
Ti64 OCRP config, changing only the local-support hyperparameters implicated
by the window-scale diagnostic.
"""

from __future__ import annotations

import json
import math
import shlex
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")
EXP_ROOT = ROOT / "experiments" / "Ti64_DIC_Mclean_fresh_4x4_existing_split"
BASE_EXP = EXP_ROOT / "ocrp_direct_reynolds_isometric_l6_4x4_fresh_ti64_dic_mclean"
NEW_EXP = EXP_ROOT / "ocrp_direct_reynolds_isometric_l6_w5local_4x4_fresh_ti64_dic_mclean"
BASE_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4"
NEW_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4_w5local"
CONFIG_NAME = "config_new.json"
GPU = 0


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def write_w5_manifest_and_postprocess() -> None:
    base_manifest_path = BASE_OUT / "manifest.json"
    if not base_manifest_path.exists():
        raise FileNotFoundError(base_manifest_path)

    manifest = read_json(base_manifest_path)
    manifest["created_by"] = Path(__file__).name
    manifest["analysis_out"] = str(NEW_OUT)
    manifest["outputs"] = {
        "metrics_csv": str(NEW_OUT / "metrics.csv"),
        "metrics_json": str(NEW_OUT / "metrics.json"),
        "metrics_md": str(NEW_OUT / "metrics.md"),
        "best_by_metric_csv": str(NEW_OUT / "best_by_metric.csv"),
    }
    manifest["ocrp_w5local_rerun"] = {
        "base_manifest": str(base_manifest_path),
        "base_ocrp_experiment": str(BASE_EXP),
        "new_ocrp_experiment": str(NEW_EXP),
    }

    new_run = {
        "name": "OCRP-direct-Reynolds-isometric-L6-w5local",
        "slug": NEW_EXP.name,
        "kind": "ocrp",
        "gpu": GPU,
        "exp_dir": str(NEW_EXP),
        "config_name": CONFIG_NAME,
        "config_path": str(NEW_EXP / CONFIG_NAME),
        "source_config": str(BASE_EXP / CONFIG_NAME),
        "checkpoint": str(NEW_EXP / "checkpoints" / "best_model.pt"),
        "inference_dir": str(NEW_EXP / "inference" / "test_best"),
        "summary": str(NEW_EXP / "inference" / "test_best" / "summary.json"),
    }
    runs = list(manifest["runs"])
    runs = [run for run in runs if run.get("slug") != NEW_EXP.name]
    runs.append(new_run)
    manifest["runs"] = runs
    manifest["tmux_scripts"] = {
        "postprocess": str(NEW_EXP / "logs" / "launch_postprocess_after_train.sh"),
        "session": "ti64_ocrp_w5local_post",
    }

    NEW_OUT.mkdir(parents=True, exist_ok=True)
    (NEW_OUT / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = NEW_OUT / "manifest.json"
    write_json(manifest_path, manifest)

    infer_cmd = shell_join(
        [
            PYTHON,
            ROOT / "inference" / "infer_iso_embedding_sr_attn.py",
            "--exp_dir",
            NEW_EXP,
            "--config",
            CONFIG_NAME,
            "--checkpoint",
            "best_model.pt",
            "--split",
            "Test",
            "--out_dir",
            NEW_EXP / "inference" / "test_best",
            "--skip_ipf",
            "--gpu_ids",
            str(GPU),
        ]
    )
    metrics_cmd = shell_join(
        [
            PYTHON,
            ROOT / "analysis" / "evaluate_ti64_dic_mclean_fresh_4x4.py",
            "--manifest",
            manifest_path,
            "--include-classical",
        ]
    )
    post = NEW_EXP / "logs" / "launch_postprocess_after_train.sh"
    post.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"mkdir -p {shlex.quote(str(NEW_OUT / 'logs'))}\n"
        "echo \"[$(date)] waiting for w5-local OCRP training\"\n"
        "while true; do\n"
        f"  if [[ -f {shlex.quote(str(NEW_EXP / 'logs' / 'train.done'))} && -f {shlex.quote(str(NEW_EXP / 'checkpoints' / 'best_model.pt'))} ]]; then\n"
        "    break\n"
        "  fi\n"
        "  sleep 300\n"
        "done\n"
        "echo \"[$(date)] training complete; running Test inference\"\n"
        f"{infer_cmd} > {shlex.quote(str(NEW_EXP / 'logs' / 'tmux_infer_test_best.log'))} 2>&1\n"
        f"touch {shlex.quote(str(NEW_EXP / 'logs' / 'infer_test_best.done'))}\n"
        "echo \"[$(date)] inference complete; running comparison metrics\"\n"
        f"{metrics_cmd} > {shlex.quote(str(NEW_OUT / 'logs' / 'final_metrics.log'))} 2>&1\n"
        f"touch {shlex.quote(str(NEW_OUT / 'logs' / 'final_metrics.done'))}\n"
        "echo \"[$(date)] postprocess complete\"\n"
    )
    post.chmod(post.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def main() -> None:
    cfg = read_json(BASE_EXP / CONFIG_NAME)

    cfg["checkpoints_dir"] = str(NEW_EXP / "checkpoints")
    cfg["batch_size"] = 5
    cfg["eval_batch_size"] = 5
    cfg["drop_last"] = True
    cfg["seed"] = 42
    cfg["save_every"] = 1
    cfg["viz_every"] = 1
    cfg["save_last_checkpoint"] = True
    cfg["save_epoch_checkpoints"] = True
    cfg["final_viz"] = True
    cfg["plot_loss_curves"] = True
    cfg["viz_ref_dir"] = "ALL"
    cfg["viz_sample_index"] = 0
    cfg["viz_sample_key"] = None

    # Window-scale fix for the much finer Ti64 DIC McLean LR topology.
    cfg["lr_conv1_kernel_size"] = 3
    cfg["window_size"] = 5
    cfg["cluster_window_size"] = 5
    cfg["cluster_threshold_deg"] = 5.0
    cfg["cluster_feature_l2_threshold"] = math.radians(5.0)

    # Keep a small equivariant HR refinement but avoid broad post-OCRP smoothing.
    cfg["use_hr_conv1"] = True
    cfg["hr_conv1_kernel_size"] = 3
    cfg["hr_conv1_residual_weight"] = 0.2
    cfg["use_hr_conv2"] = False
    cfg["hr_conv2_kernel_size"] = 3
    cfg["hr_conv2_residual_weight"] = 0.0
    cfg["use_hr_conv3"] = False
    cfg["hr_conv3_kernel_size"] = 3
    cfg["hr_conv3_residual_weight"] = 0.0

    cfg["ocrp_rerun_note"] = {
        "reason": (
            "Ti64 DIC McLean LR grains are much smaller than the Ti-Al source recipe: "
            "held-out diagnostics found median LR grain diameter 1.60 px and median "
            "24 grains in the old 9x9 OCRP window."
        ),
        "base_experiment": str(BASE_EXP),
        "changed_hyperparameters": {
            "window_size": 5,
            "cluster_window_size": 5,
            "lr_conv1_kernel_size": 3,
            "cluster_threshold_deg": 5.0,
            "cluster_feature_l2_threshold": math.radians(5.0),
            "hr_conv1_kernel_size": 3,
            "hr_conv1_residual_weight": 0.2,
            "use_hr_conv2": False,
        },
    }

    NEW_EXP.mkdir(parents=True, exist_ok=True)
    (NEW_EXP / "logs").mkdir(parents=True, exist_ok=True)
    write_json(NEW_EXP / CONFIG_NAME, cfg)

    train_cmd = shell_join(
        [
            PYTHON,
            ROOT / "training" / "train_iso_embedding_ocrp.py",
            "--exp_dir",
            NEW_EXP,
            "--config",
            CONFIG_NAME,
            "--gpu_ids",
            str(GPU),
        ]
    )
    launch = NEW_EXP / "logs" / "launch_train.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"{train_cmd} > {shlex.quote(str(NEW_EXP / 'logs' / 'tmux_train.log'))} 2>&1\n"
        f"touch {shlex.quote(str(NEW_EXP / 'logs' / 'train.done'))}\n"
    )
    launch.chmod(launch.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    notes = NEW_EXP / "RUN_NOTES.md"
    notes.write_text(
        "# Ti64 DIC McLean OCRP w5-local rerun\n\n"
        "This run keeps the existing dataset split and reruns OCRP with smaller local support.\n\n"
        "Key changes from the completed OCRP Ti64 run:\n"
        "- `window_size`: 9 -> 5\n"
        "- `cluster_window_size`: explicit 5\n"
        "- `lr_conv1_kernel_size`: 5 -> 3\n"
        "- `cluster_threshold_deg`: 2 -> 5\n"
        "- `cluster_feature_l2_threshold`: 2 degree equivalent -> 5 degree equivalent\n"
        "- HR refinement: one 3x3 residual pass at weight 0.2; second HR pass disabled\n\n"
        f"Launch script: `{launch}`\n"
        f"GPU: `{GPU}`\n"
    )
    write_w5_manifest_and_postprocess()

    print(f"Wrote {NEW_EXP / CONFIG_NAME}")
    print(f"Wrote {launch}")
    print(f"Wrote {notes}")
    print(f"Wrote {NEW_OUT / 'manifest.json'}")
    print(f"Wrote {NEW_EXP / 'logs' / 'launch_postprocess_after_train.sh'}")
    print(f"tmux session: ti64_ocrp_w5local")


if __name__ == "__main__":
    main()
