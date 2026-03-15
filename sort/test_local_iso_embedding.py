import torch

from models.local_iso_embedding_test_slow import (
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_o = build_local_iso_fcc_embedding(device=device)
    emb_d6 = build_local_iso_hcp_embedding(device=device, d6_convention="z_axis")

    print("device:", device)
    print("O irreps_out :", emb_o.irreps_out)
    print("D6 irreps_out:", emb_d6.irreps_out)

    I_o = torch.eye(3, dtype=torch.float64, device=emb_o.group_mats.device)
    I_d6 = torch.eye(3, dtype=torch.float64, device=emb_d6.group_mats.device)

    G_o_raw = emb_o.gram_at_identity(use_raw=True)
    G_d6_raw = emb_d6.gram_at_identity(use_raw=True)

    print("\nRaw local-isometry check")
    print("||G_O_raw-I||_max :", float((G_o_raw - I_o).abs().max().item()))
    print("||G_D6_raw-I||_max:", float((G_d6_raw - I_d6).abs().max().item()))

    err_o = emb_o.right_invariance_error(use_raw=True, n_trials=10)
    err_d6 = emb_d6.right_invariance_error(use_raw=True, n_trials=10)

    print("\nRight-invariance max errors (raw)")
    print(f"O  : {err_o:.3e}")
    print(f"D6 : {err_d6:.3e}")

    # Small forward smoke checks
    R = torch.eye(3, dtype=torch.float64, device=emb_o.group_mats.device).unsqueeze(0)
    print("\nFeature shapes")
    print("O raw    :", tuple(emb_o.forward_raw(R).shape))
    print("O irreps :", tuple(emb_o.forward_irreps(R).shape))

    R_d6 = torch.eye(3, dtype=torch.float64, device=emb_d6.group_mats.device).unsqueeze(0)
    print("D6 raw   :", tuple(emb_d6.forward_raw(R_d6).shape))
    print("D6 irreps:", tuple(emb_d6.forward_irreps(R_d6).shape))


if __name__ == "__main__":
    main()
