# -*-coding:utf-8 -*-
"""
File:        schedulers.py
Author:      Warren Zamudio
Description: Learning rate scheduler factory.
"""

import math
import torch


def build_scheduler(optimizer, cfg):
    """Build LR scheduler based on config."""
    sched_cfg = cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine":
        warmup_epochs = sched_cfg.get("warmup_epochs", 1)
        total_epochs = cfg["epochs"]
        min_lr = sched_cfg.get("min_lr", 1e-6)
        base_lr = cfg["lr"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine_factor)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.5),
        )

    else:
        print(f"⚠️ No scheduler found for type '{sched_type}' — using constant LR")
        return None
