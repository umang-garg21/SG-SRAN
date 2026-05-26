"""
Dedicated OCRP training entrypoint.

This reuses the shared quaternion-SR trainer and relies on the experiment
config to select `models.SR_ocrp.IsoEmbeddingSROCRP`.

We parse `--gpu_ids` here and export `CUDA_VISIBLE_DEVICES` *before* importing
the shared trainer so CUDA device masking is established ahead of any Torch/CUDA
initialization that may happen during module import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _set_cuda_visible_devices_from_argv(argv: list[str]) -> None:
    gpu_ids: str | None = None
    for idx, token in enumerate(argv):
        if token == "--gpu_ids" and idx + 1 < len(argv):
            gpu_ids = argv[idx + 1]
            break
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


_set_cuda_visible_devices_from_argv(sys.argv[1:])

from training.train_iso_embedding_sr_attn import main


if __name__ == "__main__":
    main()
