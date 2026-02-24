import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.autoencoder import FCCAutoEncoder
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export teacher table for decoder distillation")
    parser.add_argument("--exp_dir", required=True, type=str)
    parser.add_argument("--config", default="config.json", type=str)
    parser.add_argument("--out", required=True, type=str, help="Output .pt path")
    parser.add_argument("--split", default="Train", choices=["Train", "Val", "Test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--decoder_backend", default="lookup", type=str)
    parser.add_argument("--decoder_lookup_resolution", type=int, default=1)
    parser.add_argument("--decoder_lookup_npy_path", default="symmetry_groups/fast_lookup_res1.npy", type=str)
    parser.add_argument("--decoder_lookup_rebuild", action="store_true")
    parser.add_argument("--decoder_lookup_refine_steps", type=int, default=32)
    parser.add_argument("--decoder_lookup_refine_lr", type=float, default=0.001)
    parser.add_argument("--decoder_w6", type=float, default=0.5)

    parser.add_argument(
        "--include_cubochoric_fz",
        action="store_true",
        help="Also append cubochoric FZ samples from ORIX get_sample_fundamental().",
    )
    parser.add_argument(
        "--cubochoric_only",
        action="store_true",
        help="Export only cubochoric FZ samples (skip dataset rows).",
    )
    parser.add_argument(
        "--cubochoric_resolution",
        type=int,
        default=1,
        help="Resolution passed to get_sample_fundamental for cubochoric samples.",
    )
    parser.add_argument(
        "--cubochoric_method",
        type=str,
        default="cubochoric",
        help="Sampling method passed to get_sample_fundamental (default: cubochoric).",
    )
    parser.add_argument(
        "--cubochoric_max_samples",
        type=int,
        default=None,
        help="Optional cap on number of cubochoric FZ quaternions to append.",
    )
    parser.add_argument(
        "--cubochoric_chunk_size",
        type=int,
        default=8192,
        help="Chunk size for decode/match when appending cubochoric FZ samples.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32"],
        help="Dtype for saved tensors in the output .pt file (precision-locked to float32).",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    cfg = load_and_prepare_config(exp_dir / args.config, exp_dir / "logs" / "run_config_export_teacher.json")

    batch_size = int(args.batch_size) if args.batch_size is not None else int(getattr(cfg, "batch_size", 8))
    use_cubochoric = bool(args.include_cubochoric_fz or args.cubochoric_only)

    loader = None
    if not args.cubochoric_only:
        loader = build_dataloader(
            dataset_root=getattr(cfg, "dataset_root"),
            split=args.split,
            batch_size=batch_size,
            num_workers=int(getattr(cfg, "num_workers", 0)),
            preload=bool(getattr(cfg, "preload", True)),
            preload_torch=bool(getattr(cfg, "preload_torch", True)),
            pin_memory=bool(getattr(cfg, "pin_memory", True)),
            take_first=args.max_samples,
            seed=int(getattr(cfg, "seed", 42)),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = FCCAutoEncoder(
        device=device,
        decoder_backend=args.decoder_backend,
        decoder_lookup_resolution=args.decoder_lookup_resolution,
        decoder_lookup_npy_path=args.decoder_lookup_npy_path,
        decoder_lookup_rebuild=args.decoder_lookup_rebuild,
        decoder_lookup_refine_steps=args.decoder_lookup_refine_steps,
        decoder_lookup_refine_lr=args.decoder_lookup_refine_lr,
        decoder_w6=args.decoder_w6,
    )
    teacher.eval()

    f4_list: list[torch.Tensor] = []
    f6_list: list[torch.Tensor] = []
    q_teacher_list: list[torch.Tensor] = []
    q_input_list: list[torch.Tensor] = []
    rows_exported = 0

    if loader is not None:
        progress = tqdm(loader, desc="Exporting teacher table", unit="batch")
        for _, hr in progress:
            for b in range(int(hr.shape[0])):
                q = hr[b].permute(1, 2, 0).reshape(-1, 4).to(device)
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
                q, _ = teacher._reduce_to_fz(q)

                f4, f6 = teacher.encode(q)
                q_dec = teacher.decode(f4, f6)
                q_match = teacher.reduce_to_fz(q_dec)

                f4_list.append(f4.detach().cpu())
                f6_list.append(f6.detach().cpu())
                q_teacher_list.append(q_match.detach().cpu())
                q_input_list.append(q.detach().cpu())
                rows_exported += int(q.shape[0])

            progress.set_postfix(rows=rows_exported)

    rows_from_dataset = rows_exported
    rows_from_cubochoric = 0

    if use_cubochoric:
        q_cub = teacher._sample_fz_quaternions(
            resolution=int(args.cubochoric_resolution),
            method=str(args.cubochoric_method),
            device=device,
        )
        q_cub = q_cub / q_cub.norm(dim=1, keepdim=True).clamp_min(1e-12)
        q_cub, _ = teacher._reduce_to_fz(q_cub)

        if args.cubochoric_max_samples is not None:
            keep = int(args.cubochoric_max_samples)
            if keep > 0:
                q_cub = q_cub[:keep]

        chunk = max(1, int(args.cubochoric_chunk_size))
        cub_progress = tqdm(
            range(0, int(q_cub.shape[0]), chunk),
            desc="Appending cubochoric FZ",
            unit="chunk",
        )
        for start in cub_progress:
            end = min(start + chunk, int(q_cub.shape[0]))
            q_chunk = q_cub[start:end]

            f4, f6 = teacher.encode(q_chunk)
            q_dec = teacher.decode(f4, f6)
            q_match = teacher.reduce_to_fz(q_dec)

            f4_list.append(f4.detach().cpu())
            f6_list.append(f6.detach().cpu())
            q_teacher_list.append(q_match.detach().cpu())
            q_input_list.append(q_chunk.detach().cpu())

            rows_exported += int(q_chunk.shape[0])
            rows_from_cubochoric += int(q_chunk.shape[0])
            cub_progress.set_postfix(rows=rows_exported)

    if len(f4_list) == 0:
        raise ValueError(
            "No rows were exported. Use dataset rows or enable cubochoric rows via "
            "--include_cubochoric_fz / --cubochoric_only."
        )

    f4_all = torch.cat(f4_list, dim=0)
    f6_all = torch.cat(f6_list, dim=0)
    q_teacher_all = torch.cat(q_teacher_list, dim=0)
    q_input_all = torch.cat(q_input_list, dim=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dtype = torch.float32

    payload = {
        "f4": f4_all.to(save_dtype),
        "f6": f6_all.to(save_dtype),
        "q_teacher": q_teacher_all.to(save_dtype),
        "q_input": q_input_all.to(save_dtype),
        "meta": {
            "exp_dir": str(exp_dir),
            "config": args.config,
            "split": args.split,
            "decoder_backend": args.decoder_backend,
            "decoder_lookup_resolution": int(args.decoder_lookup_resolution),
            "decoder_lookup_npy_path": args.decoder_lookup_npy_path,
            "decoder_lookup_refine_steps": int(args.decoder_lookup_refine_steps),
            "decoder_lookup_refine_lr": float(args.decoder_lookup_refine_lr),
            "decoder_w6": float(args.decoder_w6),
            "include_cubochoric_fz": bool(use_cubochoric),
            "cubochoric_only": bool(args.cubochoric_only),
            "cubochoric_resolution": int(args.cubochoric_resolution),
            "cubochoric_method": str(args.cubochoric_method),
            "save_dtype": str(args.save_dtype).lower(),
            "rows_from_dataset": int(rows_from_dataset),
            "rows_from_cubochoric": int(rows_from_cubochoric),
            "num_rows": int(f4_all.shape[0]),
        },
    }
    torch.save(payload, out_path)
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"saved_teacher_table {out_path}")
    print(f"num_rows {f4_all.shape[0]}")
    print(f"file_size_mb {size_mb:.2f}")


if __name__ == "__main__":
    main()
