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
    sched_cfg = getattr(cfg, "scheduler", {})
    sched_type = getattr(sched_cfg, "type", "cosine")

    if sched_type == "cosine":
        warmup_epochs = getattr(sched_cfg, "warmup_epochs", 1)
        total_epochs = getattr(cfg, "epochs", None)
        min_lr = getattr(sched_cfg, "min_lr", 1e-6)
        base_lr = getattr(cfg, "lr", 1e-3)

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
            step_size=getattr(sched_cfg, "step_size", 10),
            gamma=getattr(sched_cfg, "gamma", 0.5),
        )

    else:
        print(f"⚠️ No scheduler found for type '{sched_type}' — using constant LR")
        return None
