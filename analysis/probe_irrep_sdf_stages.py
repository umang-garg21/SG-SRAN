#!/usr/bin/env python3
"""Decode and visualize intermediate probe stages for the modified one-sided SR model."""

from __future__ import annotations

import atexit
import argparse
import gc
import inspect
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.infer_iso_embedding_sr_attn import (
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
    _resolve_model_class,
    _to_hwc_quat_single,
    _unpack_batch,
)
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from utils.stage_probe_utils import (
    compute_attention_probe_traces,
    decode_probe_stages,
    extract_explicit_scalar_probe_maps,
    pick_most_free_cuda_gpu,
    render_attention_probe_gallery,
    render_decoded_probe_gallery,
    render_sdf_comparison,
    render_scalar_probe_gallery,
    render_upsampler_boundary_overlay,
    sample_attention_probe_pixels,
    select_upsampler_stage_row,
)
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe intermediate A1 stages for modified one-sided SR")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument("--config", type=str, default="config_new.json", help="Config file inside exp_dir.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename or absolute path.")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"], help="Dataset split.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index to visualize.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value. By default the most-free GPU is selected.",
    )
    parser.add_argument("--attn_probe_total", type=int, default=10, help="Number of random attention probe pixels.")
    parser.add_argument("--attn_probe_boundary", type=int, default=3, help="How many attention probe pixels should be on the HR boundary.")
    parser.add_argument("--attn_probe_seed", type=int, default=0, help="Random seed for probe-pixel selection.")
    parser.add_argument(
        "--no_cleanup_existing",
        action="store_true",
        help="Do not evict older probe runs targeting the same output directory.",
    )
    return parser.parse_args()


def _resolve_loader_flags(cfg, model_cls) -> tuple[bool, float, bool]:
    forward_sr_params_cls = inspect.signature(model_cls.forward_sr).parameters
    model_supports_lr_boundary = "lr_boundary_map" in forward_sr_params_cls
    model_requires_lr_boundary = (
        model_supports_lr_boundary
        and forward_sr_params_cls["lr_boundary_map"].default is inspect._empty
    )
    feature_upsampler_type = str(getattr(cfg, "feature_upsampler_type", "shifted_bilinear")).strip().lower()
    use_lr_boundary_map = bool(getattr(cfg, "use_lr_boundary_map", model_supports_lr_boundary))
    if feature_upsampler_type == "grain_attention":
        use_lr_boundary_map = True
    if model_requires_lr_boundary:
        use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    return use_lr_boundary_map, lr_boundary_angle_deg, lr_boundary_mark_both_sides


def _default_out_dir(exp_dir: Path, split: str, sample_idx: int) -> Path:
    return exp_dir / "analysis" / "stage_probe" / f"{str(split).lower()}_sample{int(sample_idx):04d}"


def _read_proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return []
    parts = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part]
    return parts


def _parse_probe_identity(argv: list[str]) -> dict[str, Path | str | int] | None:
    if not argv:
        return None
    joined = " ".join(argv)
    if "probe_irrep_sdf_stages.py" not in joined:
        return None

    parsed: dict[str, str] = {}
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token.startswith("--"):
            if "=" in token:
                key, value = token[2:].split("=", 1)
                parsed[key] = value
            elif idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                parsed[token[2:]] = argv[idx + 1]
                idx += 1
            else:
                parsed[token[2:]] = "true"
        idx += 1

    exp_dir_raw = parsed.get("exp_dir")
    if not exp_dir_raw:
        return None
    exp_dir = Path(exp_dir_raw).resolve()
    split = parsed.get("split", "Test")
    sample_idx = int(parsed.get("sample_idx", "0"))
    out_dir_raw = parsed.get("out_dir")
    out_dir = Path(out_dir_raw).resolve() if out_dir_raw else _default_out_dir(exp_dir, split, sample_idx)
    return {
        "exp_dir": exp_dir,
        "split": split,
        "sample_idx": sample_idx,
        "out_dir": out_dir,
    }


def _terminate_pid(pid: int, grace_seconds: float = 3.0) -> str:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "missing"
    except PermissionError:
        return "permission-denied"

    deadline = time.time() + float(grace_seconds)
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "terminated"
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    except PermissionError:
        return "permission-denied"
    return "killed"


class ProbeRunGuard:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.pid = os.getpid()
        self._cleaned = False
        self._registered = False

    def _cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True

        if self.lock_path.exists():
            try:
                info = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            if int(info.get("pid", -1)) == self.pid:
                self.lock_path.unlink(missing_ok=True)

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _signal_handler(self, signum, frame) -> None:  # type: ignore[override]
        self._cleanup()
        raise SystemExit(128 + int(signum))

    def register(self) -> None:
        if self._registered:
            return
        self._registered = True
        atexit.register(self._cleanup)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._signal_handler)

    def write_lock(self, payload: dict[str, object]) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cleanup_existing_probe_runs(target_out_dir: Path, keep_pid: int) -> list[dict[str, object]]:
    terminated: list[dict[str, object]] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == int(keep_pid):
            continue
        argv = _read_proc_cmdline(pid)
        info = _parse_probe_identity(argv)
        if info is None:
            continue
        if Path(info["out_dir"]) != target_out_dir:
            continue
        status = _terminate_pid(pid)
        terminated.append({"pid": pid, "status": status, "cmd": " ".join(argv)})
    return terminated


def main() -> None:
    args = parse_args()
    selected_gpu: int | None = None
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        print(f"Using requested GPU(s): {args.gpu_ids}")
    elif torch.cuda.is_available():
        selected_gpu = pick_most_free_cuda_gpu()
        if selected_gpu is not None:
            print(f"Auto-selected most-free GPU: {selected_gpu}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{selected_gpu}" if selected_gpu is not None else "cuda")
    else:
        device = torch.device("cpu")
    exp_dir = Path(args.exp_dir).resolve()
    config_path = exp_dir / args.config
    sample_idx = max(0, int(args.sample_idx))
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else _default_out_dir(exp_dir, args.split, sample_idx)
    )
    lock_path = out_dir / ".probe_run.lock.json"
    guard = ProbeRunGuard(lock_path)
    guard.register()

    cleaned_processes: list[dict[str, object]] = []
    if not bool(args.no_cleanup_existing):
        cleaned_processes = _cleanup_existing_probe_runs(out_dir, keep_pid=os.getpid())
        if cleaned_processes:
            cleaned_pids = ", ".join(str(item["pid"]) for item in cleaned_processes)
            print(f"Cleaned older probe runs for {out_dir}: {cleaned_pids}")
    guard.write_lock(
        {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "out_dir": str(out_dir),
            "exp_dir": str(exp_dir),
            "split": str(args.split),
            "sample_idx": sample_idx,
            "cleanup_existing": not bool(args.no_cleanup_existing),
            "cleaned_processes": cleaned_processes,
            "argv": sys.argv,
        }
    )

    cfg = load_and_prepare_config(config_path, exp_dir / "logs" / "probe_run_config.json")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)

    model_cls, _, _ = _resolve_model_class(cfg)
    use_lr_boundary_map, lr_boundary_angle_deg, lr_boundary_mark_both_sides = _resolve_loader_flags(cfg, model_cls)

    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=str(args.split).capitalize(),
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=sample_idx + 1,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=use_lr_boundary_map,
        lr_boundary_angle_deg=lr_boundary_angle_deg,
        lr_boundary_mark_both_sides=lr_boundary_mark_both_sides,
    )

    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_batch = None
    for idx, batch in enumerate(loader):
        if idx == sample_idx:
            selected_batch = batch
            break
    if selected_batch is None:
        raise IndexError(f"sample_idx={sample_idx} is out of range for split={args.split!r}")

    lr_batch, hr_batch, lr_boundary_batch = _unpack_batch(selected_batch)
    lr = lr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
    hr = hr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
    lr_boundary = None
    if lr_boundary_batch is not None:
        lr_boundary = lr_boundary_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)

    lr_flat, lr_shape = _flatten_quat_chw(lr)
    hr_hwc = _to_hwc_quat_single(hr).detach().cpu()
    lr_hwc = _to_hwc_quat_single(lr).detach().cpu()

    forward_kwargs = {
        "lr_shape": lr_shape,
        "normalize_input": True,
        "return_aux": True,
        "return_probe": True,
    }
    if lr_boundary is not None:
        forward_kwargs["lr_boundary_map"] = lr_boundary

    with torch.enable_grad():
        sr_flat, aux = model.forward_sr(lr_flat, **forward_kwargs)

    probe_stages = aux.get("probe_stages")
    if not isinstance(probe_stages, list) or len(probe_stages) == 0:
        raise RuntimeError("Model did not return any probe stages. Expected aux['probe_stages'].")

    decoded_stages = decode_probe_stages(model, probe_stages, sample_index=0)
    sr_h, sr_w = decoded_stages[-1]["shape"]
    sr_hwc = sr_flat.reshape(sr_h, sr_w, 4).detach().cpu()

    context_rows = [
        {"name": "lr_input", "shape": tuple(lr_hwc.shape[:2]), "quat_hwc": lr_hwc, "hr_target_hwc": hr_hwc},
        *[
            {
                "name": item["name"],
                "shape": item["shape"],
                "quat_hwc": item["quat_hwc"],
                "hr_target_hwc": hr_hwc,
            }
            for item in decoded_stages
        ],
        {"name": "sr_output", "shape": tuple(sr_hwc.shape[:2]), "quat_hwc": sr_hwc, "hr_target_hwc": hr_hwc},
        {"name": "hr_target", "shape": tuple(hr_hwc.shape[:2]), "quat_hwc": hr_hwc, "hr_target_hwc": hr_hwc},
    ]

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    decoded_gallery_path = render_decoded_probe_gallery(
        context_rows,
        sym_class=sym_class,
        out_png=out_dir / "decoded_probe_gallery.png",
    )

    upsampler_overlay_path = None
    upsampler_overlay_stage = None
    sdf_comparison_path = None
    lr_labels_dense_single = None
    sdf_hr_single = None
    hr_to_lr_owner_single = None
    boundary_lr_1px_single = None
    lr_labels_dense = aux.get("lr_labels_dense")
    if isinstance(lr_labels_dense, torch.Tensor):
        if lr_labels_dense.dim() == 3:
            lr_labels_dense_single = lr_labels_dense[0]
        elif lr_labels_dense.dim() == 2:
            lr_labels_dense_single = lr_labels_dense

    boundary_lr_1px = aux.get("boundary_lr_1px")
    if isinstance(boundary_lr_1px, torch.Tensor):
        if boundary_lr_1px.dim() == 4:
            boundary_lr_1px_single = boundary_lr_1px[0, 0]
        elif boundary_lr_1px.dim() == 3:
            boundary_lr_1px_single = boundary_lr_1px[0]
        elif boundary_lr_1px.dim() == 2:
            boundary_lr_1px_single = boundary_lr_1px

    sdf_hr = aux.get("sdf_hr")
    if isinstance(sdf_hr, torch.Tensor):
        if sdf_hr.dim() == 4:
            sdf_hr_single = sdf_hr[0, 0]
        elif sdf_hr.dim() == 3:
            sdf_hr_single = sdf_hr[0]
        elif sdf_hr.dim() == 2:
            sdf_hr_single = sdf_hr

    hr_to_lr_owner = aux.get("hr_to_lr_owner")
    if isinstance(hr_to_lr_owner, torch.Tensor):
        if hr_to_lr_owner.dim() == 3:
            hr_to_lr_owner_single = hr_to_lr_owner[0]
        elif hr_to_lr_owner.dim() == 2:
            hr_to_lr_owner_single = hr_to_lr_owner

    if lr_labels_dense_single is not None and sdf_hr_single is not None and hr_to_lr_owner_single is not None:
        try:
            upsampler_stage_row = select_upsampler_stage_row(context_rows)
            upsampler_overlay_stage = str(upsampler_stage_row["name"])
            upsampler_overlay_path = render_upsampler_boundary_overlay(
                lr_stage_row=context_rows[0],
                lr_labels_dense=lr_labels_dense_single,
                upsampler_stage_row=upsampler_stage_row,
                sdf_hr=sdf_hr_single,
                hr_to_lr_owner=hr_to_lr_owner_single,
                sym_class=sym_class,
                out_png=out_dir / "upsampler_boundary_overlay.png",
            )
        except ValueError as exc:
            print(f"Skipping upsampler boundary overlay: {exc}")

    if (
        boundary_lr_1px_single is not None
        and sdf_hr_single is not None
        and getattr(model, "grain_attention_helper", None) is not None
    ):
        try:
            helper = model.grain_attention_helper
            oldprep_sdf_hr = helper._smooth_boundary_to_sdf_like_boundary_prep(
                boundary_lr=boundary_lr_1px_single.unsqueeze(0).unsqueeze(0).to(dtype=sdf_hr_single.dtype),
                hr_shape=tuple(int(v) for v in sdf_hr_single.shape),
            )[0, 0].detach().cpu()
            sdf_comparison_path = render_sdf_comparison(
                oldprep_sdf_hr=oldprep_sdf_hr,
                learned_sdf_hr=sdf_hr_single,
                out_png=out_dir / "oldprep_sdf_vs_sdf_hr.png",
            )
        except ValueError as exc:
            print(f"Skipping SDF comparison figure: {exc}")

    scalar_maps = extract_explicit_scalar_probe_maps(aux, sample_index=0)
    scalar_gallery_path = None
    if scalar_maps:
        scalar_gallery_path = render_scalar_probe_gallery(
            scalar_maps,
            out_png=out_dir / "scalar_probe_gallery.png",
        )

    attention_gallery_path = None
    attention_probe_points = None
    if (
        str(aux.get("feature_upsampler_type", "")).strip().lower() == "grain_attention"
        and getattr(model, "grain_attention_helper", None) is not None
        and "boundary_hr_1px" in aux
    ):
        boundary_hr = aux["boundary_hr_1px"]
        if isinstance(boundary_hr, torch.Tensor) and boundary_hr.dim() == 4:
            boundary_hr = boundary_hr[0, 0]
        elif isinstance(boundary_hr, torch.Tensor) and boundary_hr.dim() == 3:
            boundary_hr = boundary_hr[0]
        attention_probe_points = sample_attention_probe_pixels(
            boundary_mask_hr=boundary_hr,
            num_total=int(args.attn_probe_total),
            num_boundary=int(args.attn_probe_boundary),
            seed=int(args.attn_probe_seed),
        )
        attention_traces = compute_attention_probe_traces(
            model_obj=model,
            aux=aux,
            probe_points=attention_probe_points,
            sample_index=0,
        )
        attention_gallery_path = render_attention_probe_gallery(
            probe_traces=attention_traces,
            sr_quat_hwc=sr_hwc,
            sym_class=sym_class,
            out_png=out_dir / "attention_probe_gallery.png",
        )

    torch.save(
        {
            "probe_stages": probe_stages,
            "decoded_stage_names": [item["name"] for item in decoded_stages],
            "lr_quat_hwc": lr_hwc,
            "sr_quat_hwc": sr_hwc,
            "hr_quat_hwc": hr_hwc,
            "scalar_probe_names": [item["name"] for item in scalar_maps],
            "attention_probe_points": attention_probe_points,
        },
        out_dir / "probe_bundle.pt",
    )

    metadata = {
        "exp_dir": str(exp_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": str(args.split),
        "sample_idx": sample_idx,
        "decoded_gallery": str(decoded_gallery_path),
        "upsampler_boundary_overlay": str(upsampler_overlay_path) if upsampler_overlay_path is not None else None,
        "upsampler_boundary_overlay_stage": upsampler_overlay_stage,
        "sdf_comparison": str(sdf_comparison_path) if sdf_comparison_path is not None else None,
        "scalar_gallery": str(scalar_gallery_path) if scalar_gallery_path is not None else None,
        "attention_gallery": str(attention_gallery_path) if attention_gallery_path is not None else None,
        "probe_stage_names": [item["name"] for item in decoded_stages],
        "scalar_probe_names": [item["name"] for item in scalar_maps],
        "attention_probe_points": attention_probe_points,
    }
    with open(out_dir / "probe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved decoded gallery: {decoded_gallery_path}")
    if upsampler_overlay_path is not None:
        print(f"Saved upsampler boundary overlay: {upsampler_overlay_path}")
    if sdf_comparison_path is not None:
        print(f"Saved SDF comparison: {sdf_comparison_path}")
    if scalar_gallery_path is not None:
        print(f"Saved scalar gallery: {scalar_gallery_path}")
    if attention_gallery_path is not None:
        print(f"Saved attention gallery: {attention_gallery_path}")
    print(f"Saved probe bundle: {out_dir / 'probe_bundle.pt'}")
    guard._cleanup()


if __name__ == "__main__":
    main()
