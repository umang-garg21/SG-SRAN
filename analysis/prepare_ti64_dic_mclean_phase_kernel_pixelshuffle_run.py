#!/usr/bin/env python3
"""Prepare the Ti64 DIC McLean phase-kernel pixel-shuffle 4x4 experiment."""

from __future__ import annotations

import json
import shlex
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/home/umang/miniconda3/envs/material/bin/python")
EXP_ROOT = ROOT / "experiments" / "Ti64_DIC_Mclean_fresh_4x4_existing_split"
BASE_EXP = EXP_ROOT / "ocrp_direct_reynolds_isometric_l6_w5local_4x4_fresh_ti64_dic_mclean"
NEW_EXP = EXP_ROOT / "phase_kernel_pixelshuffle_l6_4x4_fresh_ti64_dic_mclean"
SOURCE_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4_w5local"
FALLBACK_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4"
NEW_OUT = ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4_phase_kernel_pixelshuffle"
CONFIG_NAME = "config_new.json"
GPU = 1


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def source_manifest_path() -> Path:
    candidate = SOURCE_OUT / "manifest.json"
    if candidate.exists():
        return candidate
    return FALLBACK_OUT / "manifest.json"


def write_manifest_and_postprocess() -> None:
    base_manifest_path = source_manifest_path()
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
    manifest["phase_kernel_pixelshuffle_run"] = {
        "source_manifest": str(base_manifest_path),
        "base_config_experiment": str(BASE_EXP),
        "new_experiment": str(NEW_EXP),
        "mechanism": (
            "Sixteen independent symmetry-valid LR feature kernels, one per 4x4 "
            "subpixel phase, are pixel-shuffled into HR feature space and refined "
            "by direct equivariant HR convolutions before decoding."
        ),
    }

    new_run = {
        "name": "Phase-kernel pixel-shuffle direct-Reynolds-isometric-L6",
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
        "session": "ti64_phase_kernel_ps_post",
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
        "echo \"[$(date)] waiting for phase-kernel pixel-shuffle training\"\n"
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

    cfg["model"] = {
        "type": "phase_kernel_pixelshuffle",
        "model_module": "models.SR_phase_kernel_pixelshuffle",
        "model_class": "IsoEmbedding4x4PhaseKernelPixelShuffleSR",
    }

    cfg["use_lr_conv1"] = True
    cfg["lr_conv1_kernel_size"] = 3
    cfg["use_residual_lr1"] = True
    cfg["lr_conv1_residual_weight"] = 1.0
    cfg["conv_feature_mask_cosine_threshold"] = 0.97
    cfg["conv_feature_mask_soft"] = False
    cfg["conv_feature_mask_temperature"] = 32.0

    cfg["phase_kernel_size"] = 3
    cfg["phase_use_residual"] = True
    cfg["phase_residual_weight"] = 1.0
    cfg["phase_feature_mask_cosine_threshold"] = 0.97
    cfg["phase_feature_mask_soft"] = False
    cfg["phase_feature_mask_temperature"] = 32.0

    cfg["use_hr_conv1"] = True
    cfg["hr_conv1_kernel_size"] = 3
    cfg["hr_conv1_residual_weight"] = 0.2
    cfg["use_hr_conv2"] = True
    cfg["hr_conv2_kernel_size"] = 3
    cfg["hr_conv2_residual_weight"] = 0.2
    cfg["use_hr_conv3"] = False
    cfg["hr_conv3_kernel_size"] = 3
    cfg["hr_conv3_residual_weight"] = 0.0

    cfg["phase_kernel_pixelshuffle_note"] = {
        "reason": (
            "Routing-free diversity test: replace the single OCRP feature stream "
            "with one symmetry-valid LR convolutional kernel per 4x4 HR phase, "
            "then pixel-shuffle and apply direct HR equivariant convolutions."
        ),
        "base_experiment": str(BASE_EXP),
        "separate_model_file": str(ROOT / "models" / "SR_phase_kernel_pixelshuffle.py"),
        "key_hyperparameters": {
            "phase_kernels": 16,
            "phase_kernel_size": 3,
            "phase_residual_weight": 1.0,
            "phase_feature_mask_cosine_threshold": 0.97,
            "hr_conv_passes": 2,
            "hr_conv_kernel_size": 3,
            "hr_conv_residual_weight": 0.2,
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
        "# Ti64 DIC McLean phase-kernel pixel-shuffle 4x4 run\n\n"
        "This run keeps the existing Train/Val/Test split and material symmetry.\n\n"
        "Mechanism:\n"
        "- encode LR quaternions with the same direct-Reynolds isometric L6 feature map\n"
        "- apply 16 independent symmetry-valid LR feature kernels, one per 4x4 phase\n"
        "- pixel-shuffle those phase features into HR feature space\n"
        "- apply two direct equivariant HR 3x3 residual feature convolutions\n"
        "- decode HR features back to quaternions\n\n"
        "This is the separate routing-free model requested for testing whether feature "
        "diversity can come from phase-specific learned LR kernels rather than OCRP slots.\n\n"
        f"Launch script: `{launch}`\n"
        f"GPU: `{GPU}`\n"
    )
    write_manifest_and_postprocess()

    print(f"Wrote {NEW_EXP / CONFIG_NAME}")
    print(f"Wrote {launch}")
    print(f"Wrote {notes}")
    print(f"Wrote {NEW_OUT / 'manifest.json'}")
    print(f"Wrote {NEW_EXP / 'logs' / 'launch_postprocess_after_train.sh'}")
    print("tmux sessions: ti64_phase_kernel_ps, ti64_phase_kernel_ps_post")


if __name__ == "__main__":
    main()
