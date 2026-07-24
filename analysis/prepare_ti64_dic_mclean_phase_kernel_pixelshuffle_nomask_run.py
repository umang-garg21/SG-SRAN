#!/usr/bin/env python3
"""Prepare the Ti64 DIC McLean phase-kernel pixel-shuffle no-mask ablation."""

from __future__ import annotations

import json
import shlex
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")
EXP_ROOT = ROOT / "experiments" / "Ti64_DIC_Mclean_fresh_4x4_existing_split"
BASE_EXP = EXP_ROOT / "phase_kernel_pixelshuffle_l6_4x4_fresh_ti64_dic_mclean"
NEW_EXP = EXP_ROOT / "phase_kernel_pixelshuffle_l6_nomask_4x4_fresh_ti64_dic_mclean"
BASE_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4_phase_kernel_pixelshuffle"
NEW_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4_phase_kernel_pixelshuffle_nomask"
CONFIG_NAME = "config_new.json"
GPU = 2
NO_MASK_THRESHOLD = -1.0


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def write_manifest_and_postprocess() -> None:
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
    manifest["phase_kernel_pixelshuffle_nomask_run"] = {
        "source_manifest": str(base_manifest_path),
        "masked_baseline_experiment": str(BASE_EXP),
        "new_experiment": str(NEW_EXP),
        "ablation": (
            "Cosine-similarity neighbor filtering is disabled by setting all "
            "CosineMaskedEquivariantSpatialConv thresholds to -1.0. Spatial "
            "softmax weights and equivariant tensor products are unchanged."
        ),
    }

    new_run = {
        "name": "Phase-kernel pixel-shuffle no-cosine-mask direct-Reynolds-isometric-L6",
        "slug": NEW_EXP.name,
        "kind": "ocrp_variant",
        "gpu": GPU,
        "exp_dir": str(NEW_EXP),
        "config_name": CONFIG_NAME,
        "config_path": str(NEW_EXP / CONFIG_NAME),
        "source_config": str(BASE_EXP / CONFIG_NAME),
        "checkpoint": str(NEW_EXP / "checkpoints" / "best_model.pt"),
        "inference_dir": str(NEW_EXP / "inference" / "test_best"),
        "summary": str(NEW_EXP / "inference" / "test_best" / "summary.json"),
    }
    runs = [run for run in list(manifest["runs"]) if run.get("slug") != NEW_EXP.name]
    runs.append(new_run)
    manifest["runs"] = runs
    manifest["tmux_scripts"] = {
        "postprocess": str(NEW_EXP / "logs" / "launch_postprocess_after_train.sh"),
        "session": "ti64_phase_kernel_nomask_post",
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
        f"export CUDA_VISIBLE_DEVICES={GPU}\n"
        f"mkdir -p {shlex.quote(str(NEW_OUT / 'logs'))}\n"
        "echo \"[$(date)] waiting for no-mask phase-kernel pixel-shuffle training\"\n"
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
    cfg["model_summary_max_rows"] = 160

    # Disable all cosine-similarity neighbor gates while keeping the same
    # equivariant kernels, residual paths, and learned spatial softmax weights.
    cfg["conv_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
    cfg["phase_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
    cfg["hr_conv_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
    cfg["hr_conv2_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
    cfg["hr_conv3_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
    cfg["conv_feature_mask_soft"] = False
    cfg["phase_feature_mask_soft"] = False
    cfg["hr_conv_feature_mask_soft"] = False
    cfg["hr_conv2_feature_mask_soft"] = False
    cfg["hr_conv3_feature_mask_soft"] = False
    inherited_note = cfg.get("phase_kernel_pixelshuffle_note")
    if isinstance(inherited_note, dict):
        key_hparams = inherited_note.get("key_hyperparameters")
        if isinstance(key_hparams, dict):
            key_hparams["phase_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
            key_hparams["conv_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD
            key_hparams["hr_conv_feature_mask_cosine_threshold"] = NO_MASK_THRESHOLD

    cfg["phase_kernel_pixelshuffle_nomask_note"] = {
        "reason": (
            "Ablation requested to remove the orientation-feature similarity condition "
            "from the phase-kernel pixel-shuffle model."
        ),
        "masked_baseline_experiment": str(BASE_EXP),
        "implementation": (
            "Cosine values are clamped to [-1, 1], so threshold=-1.0 makes every "
            "neighbor pass the hard mask. The center pixel remains included as before."
        ),
        "changed_hyperparameters": {
            "conv_feature_mask_cosine_threshold": NO_MASK_THRESHOLD,
            "phase_feature_mask_cosine_threshold": NO_MASK_THRESHOLD,
            "hr_conv_feature_mask_cosine_threshold": NO_MASK_THRESHOLD,
            "hr_conv2_feature_mask_cosine_threshold": NO_MASK_THRESHOLD,
            "hr_conv3_feature_mask_cosine_threshold": NO_MASK_THRESHOLD,
        },
    }

    NEW_EXP.mkdir(parents=True, exist_ok=True)
    (NEW_EXP / "logs").mkdir(parents=True, exist_ok=True)
    write_json(NEW_EXP / CONFIG_NAME, cfg)

    train_cmd = shell_join(
        [
            PYTHON,
            ROOT / "training" / "train_iso_embedding_sr_attn.py",
            "--exp_dir",
            NEW_EXP,
            "--config",
            CONFIG_NAME,
        ]
    )
    launch = NEW_EXP / "logs" / "launch_train.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"export CUDA_VISIBLE_DEVICES={GPU}\n"
        f"{train_cmd} > {shlex.quote(str(NEW_EXP / 'logs' / 'tmux_train.log'))} 2>&1\n"
        f"touch {shlex.quote(str(NEW_EXP / 'logs' / 'train.done'))}\n"
    )
    launch.chmod(launch.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    notes = NEW_EXP / "RUN_NOTES.md"
    notes.write_text(
        "# Ti64 DIC McLean phase-kernel pixel-shuffle no-mask ablation\n\n"
        "This run keeps the existing Train/Val/Test split and model architecture, "
        "but removes the cosine-similarity neighbor condition from all equivariant "
        "spatial convolutions by setting their thresholds to -1.0.\n\n"
        "The learned 3x3 spatial weights, equivariant tensor products, residuals, "
        "and 16 phase kernels are unchanged.\n\n"
        f"Launch script: `{launch}`\n"
        f"GPU: `{GPU}`\n"
    )
    write_manifest_and_postprocess()

    print(f"Wrote {NEW_EXP / CONFIG_NAME}")
    print(f"Wrote {launch}")
    print(f"Wrote {notes}")
    print(f"Wrote {NEW_OUT / 'manifest.json'}")
    print(f"Wrote {NEW_EXP / 'logs' / 'launch_postprocess_after_train.sh'}")
    print("tmux sessions: ti64_phase_kernel_nomask, ti64_phase_kernel_nomask_post")


if __name__ == "__main__":
    main()
