# -*-coding:utf-8 -*-
"""
File:        trainer.py
Created at:  2025/10/18 14:00:04
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import torch
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, loaders, loss_fn, writer):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.writer = writer
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.device = torch.device(cfg["device"])

    def train(self):
        self.model.train()
        total_loss = 0.0

        for lr, hr in self.loaders["train"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss_fn(sr, hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.loaders["train"])
        self.scheduler.step()
        self.writer.add_scalar("Loss/Train", avg_loss, self.epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for lr, hr in self.loaders["val"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            sr = self.model(lr)
            loss = self.loss_fn(sr, hr)
            total_loss += loss.item()

        avg_val_loss = total_loss / len(self.loaders["val"])
        self.writer.add_scalar("Loss/Val", avg_val_loss, self.epoch)
        return avg_val_loss

    def maybe_save_best(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            ckpt = Path(self.cfg["checkpoints_dir"]) / "best_model.pt"
            torch.save(self.model.state_dict(), ckpt)
