import numpy as np
import torch


from dataset_builder import QuaternionDataset
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    dataset_dir = "/data/warren/materials/EBSD/IN718_2D_SR_x4"

    train_ds = QuaternionDataset(dataset_dir, split="Train")
    val_ds = QuaternionDataset(dataset_dir, split="Val")
    test_ds = QuaternionDataset(dataset_dir, split="Test")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    rho, rho_inv = make_group_tensors_from_orix(
        sym_class=train_ds.sym_class,
        num_blocks=1,  # n_feats
        device=device,
        dtype=dtype,
    )

    # Model (Reynolds-projected equivariant upsampler)
    model = UpsamplerQuaternionTransposeConv(
        kernel_size=3,
        scale=4,
        n_feats=1,
        group_tensor=rho,
        group_tensor_inv=rho_inv,
        dropout_prob=0.0,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_psnr = -1.0
