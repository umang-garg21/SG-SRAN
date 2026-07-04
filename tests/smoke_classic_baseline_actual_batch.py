"""Run one real EBSD batch through a configured classical baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_jangid_baseline import (
    _make_loader,
    _run_epoch,
    build_loss,
    build_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((Path(cfg["dataset_root"]) / "dataset_info.json").read_text())
    symmetry = metadata["symmetry"]
    model = build_model(cfg).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    expected = int(cfg["expected_trainable_params"])
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual == expected, (actual, expected)
    learning_rate = float(cfg["lr"] if args.lr is None else args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = _make_loader(cfg, "Train", int(cfg["batch_size"]), False)
    loss = _run_epoch(
        model,
        loader,
        build_loss(symmetry),
        optimizer,
        device,
        train=True,
        max_batches=args.batches,
    )
    assert torch.isfinite(torch.tensor(loss)), loss
    print(
        f"PASS {args.config}: device={device} symmetry={symmetry} "
        f"params={actual} batches={args.batches} loss={loss:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
