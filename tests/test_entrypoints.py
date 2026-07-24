from __future__ import annotations

import subprocess
import sys


def _run_help(module_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_train_entrypoint_help() -> None:
    proc = _run_help("training.train_iso_embedding_sr_attn")
    assert proc.returncode == 0
    assert "Train IsoEmbeddingSRAttn" in proc.stdout


def test_infer_entrypoint_help() -> None:
    proc = _run_help("inference.infer_iso_embedding_sr_attn")
    assert proc.returncode == 0
    assert "Inference for IsoEmbeddingSRAttn" in proc.stdout

