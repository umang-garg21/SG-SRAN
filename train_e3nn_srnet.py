"""
Symmetry-aware EBSD SR training (x4) using QuaternionDataset pairs
with stage-by-stage IPF visualization.

Dataset:
  train_ds[i][0] = LR quats (4,32,32)
  train_ds[i][1] = HR quats (4,128,128)

Rows (visualization):
  Row 1: LR input (native 32×32, NOT upsampled)
  Row 2: LR encoded -> decoded (native 32×32, NOT upsampled)
  Row 3: HR ground truth (native 128×128)
  Row 4: SR prediction (native 128×128)

Training:
  Train SRNet in invariant feature space:
    F_lr = Enc(q_lr)
    F_hr = Enc(q_hr)
    F_pred = SRNet(F_lr)
    loss = MSE(F_pred, F_hr)
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from visualization.ipf_render import render_ipf_rgb
from orix.crystal_map import Phase

from training.data_loading import QuaternionDataset
from e3nn_sr import FCCPhysics, FCCEncoder, SphericalSamplingDecoder


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def quat_normalize_chfirst(q, eps=1e-8):
    return q / (q.norm(dim=1, keepdim=True) + eps)


def match_symmetry_batch(q_truth, q_reconstructed, physics):
    """
    q_truth, q_reconstructed: (N,4) flattened (wxyz)
    Match q_reconstructed to closest symmetry variant to best match q_truth.
    """
    B = q_truth.shape[0]
    device = q_truth.device

    q_rec_exp = q_reconstructed.unsqueeze(1).expand(-1, 24, -1)
    syms_exp = physics.fcc_syms.unsqueeze(0).expand(B, -1, -1)

    w1, x1, y1, z1 = (
        q_rec_exp[..., 0],
        q_rec_exp[..., 1],
        q_rec_exp[..., 2],
        q_rec_exp[..., 3],
    )
    w2, x2, y2, z2 = (
        syms_exp[..., 0],
        syms_exp[..., 1],
        syms_exp[..., 2],
        syms_exp[..., 3],
    )

    family = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    q_t = q_truth.unsqueeze(1)
    dist_pos = torch.norm(family - q_t, dim=-1)
    dist_neg = torch.norm(family + q_t, dim=-1)
    min_dist = torch.minimum(dist_pos, dist_neg)
    best = torch.argmin(min_dist, dim=1)

    idx = torch.arange(B, device=device)
    closest = family[idx, best]
    use_neg = dist_neg[idx, best] < dist_pos[idx, best]
    closest[use_neg] = -closest[use_neg]
    return closest


# ------------------------------------------------------------------------------
# Mixed-resolution renderer (LR rows stay LR; HR rows stay HR)
# ------------------------------------------------------------------------------
def render_lr_hr_sr_panel(rows, out_png, fcc_sym, title=None):
    """
    rows: list of dicts:
      {'name': str, 'q_chfirst': torch.Tensor shape (1,4,H,W) or (4,H,W)}
    Each row can have its own H,W.
    Produces a single figure with 4 columns: IPF-X, IPF-Y, IPF-Z, IPF key.
    """
    directions = ["X", "Y", "Z"]
    rgb_rows = []
    names = []

    for row in rows:
        q = row["q_chfirst"]
        if q.dim() == 4:
            q = q[0]
        q_img = q.detach().cpu().permute(1, 2, 0).numpy()  # (H,W,4)
        rgb = render_ipf_rgb(q_img, fcc_sym, ref_dir="ALL")  # [rgbX,rgbY,rgbZ]
        rgb_rows.append(rgb)
        names.append(row["name"])

    fig = plt.figure(figsize=(17, 4 + len(rows) * 4.2), facecolor="white")
    gs = GridSpec(
        len(rows),
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.35],
        hspace=0.25,
        wspace=0.05,
        left=0.12,
        right=0.95,
        top=0.92,
        bottom=0.05,
    )

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    for r, (name, rgb_list) in enumerate(zip(names, rgb_rows)):
        for c, (direction, rgb) in enumerate(zip(directions, rgb_list)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(rgb, interpolation="nearest")  # preserves pixel structure
            ax.axis("off")
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"IPF-{direction}", fontsize=14, fontweight="bold", pad=10)
            if c == 0:
                ax.text(
                    -0.25,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )

    ax_key = fig.add_subplot(gs[:, 3], projection="ipf", symmetry=fcc_sym.laue)
    ax_key.plot_ipf_color_key()
    ax_key.set_title("IPF Color Key", fontsize=12, fontweight="bold", pad=10)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")


# ------------------------------------------------------------------------------
# SRNet (feature-space x4)
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.a = nn.SiLU()
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(self.a(self.c1(x)))


class FeatureSRNetX4(nn.Module):
    def __init__(self, in_ch=22, hidden=128, blocks=12, scale=4):
        super().__init__()
        assert scale == 4
        self.head = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(hidden) for _ in range(blocks)])
        self.tail = nn.Conv2d(hidden, in_ch * (scale**2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.ps(self.tail(x))
        return x


# ------------------------------------------------------------------------------
# Feature encode/decode helpers
# ------------------------------------------------------------------------------
@torch.no_grad()
def encode_quat_map_to_features(encoder, q_chfirst):
    """
    q_chfirst: (B,4,H,W)
    returns: F (B,22,H,W)
    """
    B, C, H, W = q_chfirst.shape
    q_flat = q_chfirst.permute(0, 2, 3, 1).reshape(-1, 4)
    f4, f6 = encoder(q_flat)
    Fm = (
        torch.cat([f4, f6], dim=1).reshape(B, H, W, 22).permute(0, 3, 1, 2).contiguous()
    )
    return Fm


@torch.no_grad()
def decode_features_to_quat_map(decoder, F):
    """
    F: (B,22,H,W)
    returns q: (B,4,H,W)
    """
    B, _, H, W = F.shape
    f = F.permute(0, 2, 3, 1).reshape(-1, 22)
    q_flat = decoder(f[:, :9], f[:, 9:], img_shape=(B, H, W))
    return q_flat


# ------------------------------------------------------------------------------
# Training loop (dataset)
# ------------------------------------------------------------------------------
def train_srnet(
    srnet,
    encoder,
    loader,
    device,
    epochs=5,
    lr=2e-3,
    log_interval=50,
    out_dir="./training_outputs",
):
    srnet.train()
    opt = optim.Adam(srnet.parameters(), lr=lr)

    steps_per_epoch = 1273
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=steps_per_epoch, T_mult=2, eta_min=lr * 0.01
    )

    mse = nn.MSELoss()
    history = {"loss": [], "lr": []}
    t0 = time.time()
    step = 0

    for ep in range(epochs):
        for q_lr, q_hr in loader:
            q_lr = quat_normalize_chfirst(q_lr.to(device).float())
            q_hr = quat_normalize_chfirst(q_hr.to(device).float())

            with torch.no_grad():
                F_lr = encode_quat_map_to_features(encoder, q_lr)  # (B,22,32,32)
                F_hr = encode_quat_map_to_features(encoder, q_hr)  # (B,22,128,128)

            F_pred = srnet(F_lr)
            loss = mse(F_pred, F_hr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(srnet.parameters(), 1.0)
            opt.step()

            # step-based scheduler
            scheduler.step(step)

            history["loss"].append(loss.item())
            history["lr"].append(opt.param_groups[0]["lr"])

            if step % log_interval == 0:
                print(
                    f"ep {ep+1}/{epochs}  step {step:5d}  loss={loss.item():.4e}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  t={time.time()-t0:.1f}s"
                )
            step += 1

    # curves
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["loss"])
    ax[0].set_yscale("log")
    ax[0].set_title("Feature MSE")
    ax[0].grid(alpha=0.3)

    ax[1].plot(history["lr"])
    ax[1].set_yscale("log")
    ax[1].set_title("LR (CosineWarmRestarts)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"✓ Saved: {os.path.join(out_dir, 'training_curves.png')}")

    return history


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SYMMETRY-AWARE FEATURE SR (QuaternionDataset) + LR/HR STAGE IPF VIS")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_out_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4"
    dataset_dir = os.path.join(dataset_out_root, dataset_name)

    train_ds = QuaternionDataset(
        dataset_root=dataset_dir, split="Train", preload=True, preload_torch=True
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, drop_last=True, num_workers=0
    )

    out_dir = "./training_outputs"
    os.makedirs(out_dir, exist_ok=True)

    physics = FCCPhysics(device=device)
    encoder = FCCEncoder(physics).to(device)
    decoder = SphericalSamplingDecoder(physics, n_fib_samples=10000).to(device)
    srnet = FeatureSRNetX4(in_ch=22, hidden=128, blocks=12, scale=4).to(device)

    fcc_sym = Phase(space_group=225).point_group

    # sample for visualization
    q_lr0, q_hr0 = train_ds[0]
    q_lr0 = quat_normalize_chfirst(q_lr0.unsqueeze(0).to(device).float())  # (1,4,32,32)
    q_hr0 = quat_normalize_chfirst(
        q_hr0.unsqueeze(0).to(device).float()
    )  # (1,4,128,128)

    # ---------------- BEFORE ----------------
    encoder.eval()
    decoder.eval()
    srnet.eval()
    with torch.no_grad():
        # Row 2: LR enc->dec (native LR)
        F_lr0 = encode_quat_map_to_features(encoder, q_lr0)  # (1,22,32,32)
        q_lr_dec = decode_features_to_quat_map(decoder, F_lr0)  # (1,4,32,32)

        # Row 4: SR pred before training (HR)
        F_pred0 = srnet(F_lr0)  # (1,22,128,128)
        q_sr0 = decode_features_to_quat_map(decoder, F_pred0)  # (1,4,128,128)

        # Symmetry-match SR to HR truth for consistent colors (only HR-sized)
        q_hr_flat = q_hr0.permute(0, 2, 3, 1).reshape(-1, 4)
        q_sr0_flat = q_sr0.permute(0, 2, 3, 1).reshape(-1, 4)
        q_sr0_flat = match_symmetry_batch(q_hr_flat, q_sr0_flat, physics)
        q_sr0 = q_sr0_flat.reshape(1, 128, 128, 4).permute(0, 3, 1, 2).contiguous()

    rows_before = [
        {"name": "LR input (32×32)", "q_chfirst": q_lr0},
        {"name": "LR Enc→Dec (32×32)", "q_chfirst": q_lr_dec},
        {"name": "HR truth (128×128)", "q_chfirst": q_hr0},
        {"name": "SR pred BEFORE (128×128)", "q_chfirst": q_sr0},
    ]
    render_lr_hr_sr_panel(
        rows_before,
        os.path.join(out_dir, "panel_BEFORE.png"),
        fcc_sym,
        title="Before Training",
    )

    # ---------------- TRAIN ----------------
    history = train_srnet(
        srnet,
        encoder,
        loader,
        device,
        epochs=10,
        lr=2e-4,
        log_interval=25,
        out_dir=out_dir,
    )

    # ---------------- AFTER ----------------
    srnet.eval()
    with torch.no_grad():
        F_lr0 = encode_quat_map_to_features(encoder, q_lr0)
        F_pred = srnet(F_lr0)
        q_sr = decode_features_to_quat_map(decoder, F_pred)

        q_sr_flat = q_sr.permute(0, 2, 3, 1).reshape(-1, 4)
        q_sr_flat = match_symmetry_batch(q_hr_flat, q_sr_flat, physics)
        q_sr = q_sr_flat.reshape(1, 128, 128, 4).permute(0, 3, 1, 2).contiguous()

    rows_after = [
        {"name": "LR input (32x32)", "q_chfirst": q_lr0},
        {"name": "LR Enc→Dec (32x32)", "q_chfirst": q_lr_dec},
        {"name": "HR truth (128x128)", "q_chfirst": q_hr0},
        {"name": "SR pred AFTER (128x128)", "q_chfirst": q_sr},
    ]
    render_lr_hr_sr_panel(
        rows_after,
        os.path.join(out_dir, "panel_AFTER.png"),
        fcc_sym,
        title="After Training",
    )

    torch.save(srnet.state_dict(), os.path.join(out_dir, "srnet_features_x4.pt"))
    print(f"✓ Saved: {os.path.join(out_dir, 'srnet_features_x4.pt')}")

    # ------------------------------------------------------------------
    # Pick ONE test example
    # ------------------------------------------------------------------
    idx = 0

    test_ds = QuaternionDataset(
        dataset_root=dataset_dir, split="Test", preload=True, preload_torch=True
    )

    q_lr, q_hr = test_ds[idx]

    q_lr = q_lr.unsqueeze(0).to(device).float()  # (1,4,32,32)
    q_hr = q_hr.unsqueeze(0).to(device).float()  # (1,4,128,128)

    q_lr = quat_normalize_chfirst(q_lr)
    q_hr = quat_normalize_chfirst(q_hr)

    # ------------------------------------------------------------------
    # Forward passes (no gradients)
    # ------------------------------------------------------------------
    with torch.no_grad():

        # ---- Stage 1: LR → Encode → Decode
        F_lr = encode_quat_map_to_features(encoder, q_lr)
        q_lr_dec = decode_features_to_quat_map(decoder, F_lr)

        # ---- Stage 2: SR prediction
        F_sr = srnet(F_lr)
        q_sr = decode_features_to_quat_map(decoder, F_sr)

    # ------------------------------------------------------------------
    # Assemble visualization rows
    # ------------------------------------------------------------------
    rows = [
        {"name": "LR input (32×32)", "q_chfirst": q_lr},
        {"name": "LR Enc→Dec", "q_chfirst": q_lr_dec},
        {"name": "HR ground truth", "q_chfirst": q_hr},
        {"name": "SR prediction (×4)", "q_chfirst": q_sr},
    ]

    render_lr_hr_sr_panel(
        rows,
        out_png=os.path.join(out_dir, "test_panel_AFTER.png"),
        fcc_sym=fcc_sym,
        title="Test set — symmetry-aware SR",
    )

    print("✓ Saved test_panel_AFTER.png")

    print("Done.")


if __name__ == "__main__":
    main()
