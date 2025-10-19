# -*- coding:utf-8 -*-
"""
Quaternion Super-Resolution with Reynolds Projection (QSR)
Dataset: QuaternionDataset (from dataset_builder.py)
Author: Warren Zamudio
"""

import os, json, math, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from light import UpsamplerQuaternionTransposeConv
from dataset_builder import QuaternionDataset, render_ipf_image


# import model + utils from your existing file (psnr_from_mse, _hemisphere_align, _quat_loss_L1, etc.)
# e.g. UpsamplerQuaternionTransposeConv, make_group_tensors_from_orix, show_sr_vs_gt_quat


# =========================================================
# Utilities: PSNR + quaternion helpers + visualization
# =========================================================


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val**2) / mse)


def _repeat_quat_blocks(x: torch.Tensor, n_feats: int) -> torch.Tensor:
    return x if n_feats == 1 else x.repeat(1, n_feats, 1, 1)


def _hemisphere_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    d = (pred * gt).sum(dim=-1, keepdim=True)
    sign = torch.where(
        d < 0,
        torch.tensor(-1.0, device=pred.device, dtype=pred.dtype),
        torch.tensor(1.0, device=pred.device, dtype=pred.dtype),
    )
    return pred * sign


def _quat_loss_L1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs().mean()


def quat_to_C(q: np.ndarray) -> np.ndarray:
    """
    Quaternion -> 4x4 real representation matrix.
    q: (4,) scalar-first (a,b,c,d)
    """
    a, b, c, d = q
    return np.array(
        [
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ],
        dtype=np.float32,
    )


def make_group_tensors_from_orix(sym_class, num_blocks: int, device=None, dtype=None):
    """
    sym_class: e.g. ds.sym_class = orix.quaternion.symmetry.Oh
    num_blocks: number of quaternion blocks (n_feats)
    Returns rho, rho_inv: (G, 4*num_blocks, 4*num_blocks)
    """
    quats = np.array(sym_class.data)  # (G,4), scalar-first
    mats = np.stack([quat_to_C(q) for q in quats], axis=0)  # (G,4,4)

    base = torch.tensor(mats, device=device, dtype=dtype)  # (G,4,4)
    I = torch.eye(num_blocks, device=base.device, dtype=base.dtype)
    rho = torch.kron(I, base)  # (G, 4B, 4B)
    rho_inv = rho.transpose(1, 2)  # inverse = transpose (orthogonal)
    return rho, rho_inv


# =========================================================
# Training
# =========================================================


def train_with_config(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    os.makedirs(cfg["save"], exist_ok=True)

    # ---------------- Dataset ----------------
    dataset_dir = cfg["dataset_dir"]
    train_set = QuaternionDataset(dataset_dir, split="Train")
    val_set = QuaternionDataset(dataset_dir, split="Val")

    if cfg.get("train_count"):
        train_set.pairs = train_set.pairs[: cfg["train_count"]]
    if cfg.get("val_count"):
        val_set.pairs = val_set.pairs[: cfg["val_count"]]

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

    # ---------------- Group reps (Reynolds) ----------------
    rho, rho_inv = make_group_tensors_from_orix(
        sym_class=train_set.sym_class,
        num_blocks=cfg["n_feats"],
        device=device,
        dtype=dtype,
    )

    # ---------------- Model ----------------
    model = UpsamplerQuaternionTransposeConv(
        kernel_size=cfg["kernel_size"],
        scale=cfg["scale"],
        n_feats=cfg["n_feats"],
        group_tensor=rho,
        group_tensor_inv=rho_inv,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    best_psnr = -1.0
    vis_every = int(cfg.get("vis_every", 1))

    print(
        f"Training on {len(train_set)} samples, validating on {len(val_set)} samples."
    )
    for epoch in range(1, cfg["epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        total_loss = 0.0
        for LR_q, HR_q in train_loader:
            LR_q = LR_q.to(device, dtype=dtype)
            HR_q = HR_q.to(device, dtype=dtype)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                SR_q = model(_repeat_quat_blocks(LR_q, cfg["n_feats"]))
                SR_q_aligned = _hemisphere_align(
                    SR_q.permute(0, 2, 3, 1), HR_q.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)
                loss = _quat_loss_L1(SR_q_aligned, HR_q)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if cfg["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * LR_q.size(0)

        train_loss = total_loss / len(train_set)
        sched.step()

        # ---------------- Validation ----------------
        model.eval()
        mse_sum, n_elems = 0.0, 0
        first_pair = None
        with torch.no_grad():
            for LR_q, HR_q in val_loader:
                LR_q = LR_q.to(device, dtype=dtype)
                HR_q = HR_q.to(device, dtype=dtype)
                SR_q = model(_repeat_quat_blocks(LR_q, cfg["n_feats"])).clamp(-1.0, 1.0)
                SR_q_aligned = _hemisphere_align(
                    SR_q.permute(0, 2, 3, 1), HR_q.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)

                if first_pair is None:
                    first_pair = (
                        HR_q[0].cpu().permute(1, 2, 0).numpy(),
                        SR_q_aligned[0].cpu().permute(1, 2, 0).numpy(),
                    )

                SR_01, HR_01 = (SR_q_aligned + 1.0) * 0.5, (HR_q + 1.0) * 0.5
                mse_sum += F.mse_loss(SR_01, HR_01, reduction="sum").item()
                n_elems += int(np.prod(HR_01.shape))

        psnr = psnr_from_mse(mse_sum / n_elems)
        print(f"Epoch {epoch:03d} | Train L1: {train_loss:.4f} | PSNR: {psnr:.2f} dB")

        # ---- Save IPF maps every vis_every ----
        if (epoch % vis_every == 0) and first_pair is not None:
            hr_arr, sr_arr = first_pair  # (H,W,4)
            out_dir = os.path.join(cfg["save"], "ipf_val")
            os.makedirs(out_dir, exist_ok=True)

            render_ipf_image(
                hr_arr,
                train_set.sym_class,
                out_png=os.path.join(out_dir, f"epoch{epoch:03d}_HR.png"),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
            )

            render_ipf_image(
                sr_arr,
                train_set.sym_class,
                out_png=os.path.join(out_dir, f"epoch{epoch:03d}_SR.png"),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
            )

            print(f"  Saved IPF visualizations to {out_dir}")

        if psnr > best_psnr:
            best_psnr = psnr
            ckpt = os.path.join(
                cfg["save"],
                f"{train_set.info.get("dataset")}_best_model_x{cfg['scale']}.pt",
            )
            torch.save({"model": model.state_dict(), "config": cfg, "psnr": psnr}, ckpt)
            print(f"  Saved checkpoint -> {ckpt}")

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)

    # Required
    cfg.setdefault("dataset_dir", "/data/warren/materials/EBSD/IN718_2D_SR_x4")

    # Training defaults
    cfg.setdefault("epochs", 1)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("scale", 4)
    cfg.setdefault("n_feats", 1)
    cfg.setdefault("kernel_size", 3)
    cfg.setdefault("group", "Oh")
    cfg.setdefault("clip", 1.0)
    cfg.setdefault("amp", True)
    cfg.setdefault("save", "checkpoints")

    cfg.setdefault("train_count", None)
    cfg.setdefault("val_count", 200)
    cfg.setdefault("symmetry", "Oh")

    # train_with_config(cfg)

    ds = QuaternionDataset(cfg["dataset_dir"], split="Train")
