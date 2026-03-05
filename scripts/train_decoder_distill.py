import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.autoencoder import FCCPhysics
from models.autoencoder_learnable import LearnableFCCDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learnable decoder from teacher table")
    parser.add_argument("--teacher_table", required=True, type=str, help="Path to exported teacher .pt")
    parser.add_argument("--out", required=True, type=str, help="Output checkpoint path")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Optional decoder checkpoint (.pt) to initialize weights before training",
    )
    return parser.parse_args()


class DotLoss(nn.Module):
    def forward(self, q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
        dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
        return (1.0 - dots).mean()


def misorientation_deg(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dots) * 180.0 / torch.pi


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=1)


def normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=1, keepdim=True).clamp_min(eps)


def reduce_to_fz_canonical(
    q_bunge: torch.Tensor,
    fcc_syms: torch.Tensor,
    chunk_size: int = 262_144,
) -> torch.Tensor:
    q = normalize_quat(q_bunge)
    total = q.shape[0]
    step = max(1, int(chunk_size))
    q_fz_chunks: list[torch.Tensor] = []

    for start in range(0, total, step):
        end = min(start + step, total)
        q_chunk = q[start:end]
        batch_size = q_chunk.shape[0]

        q_expanded = q_chunk.unsqueeze(1).expand(-1, 24, -1)
        syms = fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)

        fam = quat_mul(
            q_expanded.reshape(-1, 4),
            syms.reshape(-1, 4),
        ).view(batch_size, 24, 4)
        fam = normalize_quat(fam.reshape(-1, 4)).view(batch_size, 24, 4)

        w_abs = fam[..., 0].abs()
        best_idx = torch.argmax(w_abs, dim=1)
        bidx = torch.arange(batch_size, device=q_chunk.device)
        q_fz = fam[bidx, best_idx]
        q_fz = torch.where(q_fz[:, :1] < 0, -q_fz, q_fz)
        q_fz_chunks.append(normalize_quat(q_fz))

    return torch.cat(q_fz_chunks, dim=0)


def reduce_to_fz_against_reference(
    q_pred_bunge: torch.Tensor,
    q_ref_bunge: torch.Tensor,
    fcc_syms: torch.Tensor,
) -> torch.Tensor:
    q_pred = normalize_quat(q_pred_bunge)
    q_ref = normalize_quat(q_ref_bunge)

    batch_size = q_ref.shape[0]
    q_exp = q_pred.unsqueeze(1).expand(-1, 24, -1)
    g = fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)
    g_inv = quat_conjugate(g.reshape(-1, 4)).view(batch_size, 24, 4)

    q_flat = q_exp.reshape(-1, 4)
    g_flat = g.reshape(-1, 4)
    g_inv_flat = g_inv.reshape(-1, 4)

    fam_right = quat_mul(q_flat, g_flat).view(batch_size, 24, 4)
    fam_left = quat_mul(g_flat, q_flat).view(batch_size, 24, 4)
    fam_right_inv = quat_mul(q_flat, g_inv_flat).view(batch_size, 24, 4)
    fam_left_inv = quat_mul(g_inv_flat, q_flat).view(batch_size, 24, 4)

    families = torch.cat([fam_right, fam_left, fam_right_inv, fam_left_inv], dim=1)
    families = normalize_quat(families.reshape(-1, 4)).view(batch_size, -1, 4)

    ref_exp = q_ref.unsqueeze(1).expand(-1, families.shape[1], -1)
    delta = quat_mul(
        families.reshape(-1, 4),
        quat_conjugate(ref_exp.reshape(-1, 4)),
    ).view(batch_size, families.shape[1], 4)
    delta = normalize_quat(delta.reshape(-1, 4)).view(batch_size, families.shape[1], 4)

    w_abs = delta[..., 0].abs().clamp(max=1.0)
    mis = 2.0 * torch.acos(w_abs)
    best_idx = torch.argmin(mis, dim=1)
    bidx = torch.arange(batch_size, device=q_pred.device)
    return families[bidx, best_idx]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        blob = torch.load(args.teacher_table, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(args.teacher_table, map_location="cpu")
    f4 = blob["f4"].to(torch.float32)
    f6 = blob["f6"].to(torch.float32)
    target_key = None
    if "q_teacher" in blob:
        q_tgt = blob["q_teacher"].to(torch.float32)
        target_key = "q_teacher"
    elif "q_input" in blob:
        q_tgt = blob["q_input"].to(torch.float32)
        target_key = "q_input"
    elif "q_input_bunge" in blob:
        q_tgt = blob["q_input_bunge"].to(torch.float32)
        target_key = "q_input_bunge"
    else:
        raise ValueError(
            "Teacher table is missing `q_teacher` and fallback keys (`q_input` or legacy `q_input_bunge`). Re-export teacher table."
        )
    q_tgt = normalize_quat(q_tgt)

    fcc_syms_cpu = FCCPhysics(device="cpu").fcc_syms.detach().to(torch.float32)
    q_tgt = reduce_to_fz_canonical(q_tgt, fcc_syms_cpu)

    n = f4.shape[0]
    perm = torch.randperm(n)
    val_n = max(1, int(n * float(args.val_ratio)))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    tr_ds = TensorDataset(f4[tr_idx], f6[tr_idx], q_tgt[tr_idx])
    va_ds = TensorDataset(f4[val_idx], f6[val_idx], q_tgt[val_idx])

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableFCCDecoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if args.init_ckpt is not None and str(args.init_ckpt).strip() != "":
        try:
            init_blob = torch.load(str(args.init_ckpt), map_location=device, weights_only=True)
        except TypeError:
            init_blob = torch.load(str(args.init_ckpt), map_location=device)

        if isinstance(init_blob, dict) and "decoder_state_dict" in init_blob:
            init_state_dict = init_blob["decoder_state_dict"]
        elif isinstance(init_blob, dict):
            init_state_dict = init_blob
        else:
            raise ValueError(f"Unsupported init checkpoint format: {args.init_ckpt}")

        init_result = model.load_state_dict(init_state_dict, strict=False)
        if hasattr(init_result, "missing_keys") and len(init_result.missing_keys) > 0:
            print(f"[distill] init_ckpt missing keys: {init_result.missing_keys[:8]}")
        if hasattr(init_result, "unexpected_keys") and len(init_result.unexpected_keys) > 0:
            print(f"[distill] init_ckpt unexpected keys: {init_result.unexpected_keys[:8]}")
        print(f"loaded_init_ckpt {args.init_ckpt}")

    fcc_syms = FCCPhysics(device=str(device)).fcc_syms.detach()

    criterion = DotLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0
        for f4_b, f6_b, q_b in tr_loader:
            f4_b = f4_b.to(device, non_blocking=True)
            f6_b = f6_b.to(device, non_blocking=True)
            q_b = q_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            q_pred = model(f4_b, f6_b)
            q_pred_fz = reduce_to_fz_against_reference(q_pred, q_b, fcc_syms)
            loss = criterion(q_pred_fz, q_b)
            loss.backward()
            optimizer.step()

            tr_loss_sum += float(loss.item()) * f4_b.shape[0]
            tr_count += int(f4_b.shape[0])

        model.eval()
        va_loss_sum = 0.0
        va_count = 0
        all_mis = []
        with torch.no_grad():
            for f4_b, f6_b, q_b in va_loader:
                f4_b = f4_b.to(device, non_blocking=True)
                f6_b = f6_b.to(device, non_blocking=True)
                q_b = q_b.to(device, non_blocking=True)
                q_pred = model(f4_b, f6_b)
                q_pred_fz = reduce_to_fz_against_reference(q_pred, q_b, fcc_syms)
                vloss = criterion(q_pred_fz, q_b)

                va_loss_sum += float(vloss.item()) * f4_b.shape[0]
                va_count += int(f4_b.shape[0])
                all_mis.append(misorientation_deg(q_pred_fz, q_b).detach().cpu())

        tr_loss = tr_loss_sum / max(1, tr_count)
        va_loss = va_loss_sum / max(1, va_count)
        mis = torch.cat(all_mis) if len(all_mis) else torch.empty(0)
        mis_mean = float(mis.mean().item()) if mis.numel() else float("nan")
        mis_max = float(mis.max().item()) if mis.numel() else float("nan")
        if mis.numel():
            qs = torch.quantile(mis, torch.tensor([0.90, 0.95, 0.99], dtype=mis.dtype))
            mis_p90 = float(qs[0].item())
            mis_p95 = float(qs[1].item())
            mis_p99 = float(qs[2].item())
        else:
            mis_p90 = float("nan")
            mis_p95 = float("nan")
            mis_p99 = float("nan")

        print(
            f"epoch {epoch + 1:03d}/{args.epochs} "
            f"train_loss={tr_loss:.6e} val_loss={va_loss:.6e} "
            f"val_mis_mean_deg={mis_mean:.4f} val_mis_p90_deg={mis_p90:.4f} "
            f"val_mis_p95_deg={mis_p95:.4f} val_mis_p99_deg={mis_p99:.4f} "
            f"val_mis_max_deg={mis_max:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                "decoder_state_dict": model.state_dict(),
                "meta": {
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                    "teacher_table": str(args.teacher_table),
                    "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
                    "target_key": str(target_key),
                    "best_val_loss": float(best_val),
                },
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {
            "decoder_state_dict": model.state_dict(),
            "meta": {
                "teacher_table": str(args.teacher_table),
                "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
                "target_key": str(target_key),
            },
        }
    torch.save(best_state, out_path)
    print(f"saved_decoder_ckpt {out_path}")


if __name__ == "__main__":
    main()
