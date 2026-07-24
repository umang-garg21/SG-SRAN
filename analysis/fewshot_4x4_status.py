#!/usr/bin/env python3
"""Summarize few-shot 4x4 adaptation job status."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis/out/fewshot_4x4_manifest.json"


def _tail(path: Path, n: int = 3) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return " | ".join(lines[-n:])


def _has_failure(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    markers = ("Traceback (most recent call last)", "RuntimeError:", "ValueError:", "FileNotFoundError:")
    return any(marker in text for marker in markers)


def _contains(path: Path, marker: str) -> bool:
    return path.exists() and marker in path.read_text(errors="replace")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for run in manifest["runs"]:
        exp_dir = ROOT / str(run["experiment"])
        best = exp_dir / "checkpoints/best_model.pt"
        summary = exp_dir / "inference/test_best/summary.json"
        train_log = exp_dir / "logs/fewshot_tmux_train.log"
        infer_log = exp_dir / "logs/fewshot_tmux_infer.log"
        train_done = (exp_dir / "logs/fewshot_train.done").exists() or _contains(
            train_log, "Training complete"
        )
        status = "pending"
        if train_log.exists():
            status = "training"
        if _has_failure(train_log):
            status = "train failed"
        if best.exists() and train_done:
            status = "trained"
        if infer_log.exists() and best.exists():
            status = "inferencing"
        if _has_failure(infer_log):
            status = "infer failed"
        if summary.exists():
            status = "inferred"
        rows.append(
            {
                "target": run["target_name"],
                "method": run["method"],
                "train": run["adaptation_samples"],
                "test": run["heldout_samples"],
                "status": status,
                "train_tail": _tail(train_log, 1),
                "infer_tail": _tail(infer_log, 1),
            }
        )

    widths = {
        "target": max(len("target"), *(len(str(row["target"])) for row in rows)),
        "method": max(len("method"), *(len(str(row["method"])) for row in rows)),
        "status": max(len("status"), *(len(str(row["status"])) for row in rows)),
    }
    print(
        f"{'target':<{widths['target']}}  "
        f"{'method':<{widths['method']}}  "
        f"{'train':>5}  {'test':>5}  "
        f"{'status':<{widths['status']}}"
    )
    print("-" * (widths["target"] + widths["method"] + widths["status"] + 24))
    for row in rows:
        print(
            f"{row['target']:<{widths['target']}}  "
            f"{row['method']:<{widths['method']}}  "
            f"{int(row['train']):>5}  {int(row['test']):>5}  "
            f"{row['status']:<{widths['status']}}"
        )


if __name__ == "__main__":
    main()
