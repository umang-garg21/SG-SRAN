import torch


def build_optimizer(model, cfg):
    opt_cfg = cfg["optimizer"]
    if opt_cfg["type"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=getattr(opt_cfg, "weight_decay", 0),
        )
    elif opt_cfg["type"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")


from training.schedulers import build_scheduler
