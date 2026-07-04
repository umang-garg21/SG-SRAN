#!/usr/bin/env python3
"""Run the 4x4 patch-comparison notebook with direct-Reynolds OCRP.

The notebook remains the source of the plotting/evaluation logic. This wrapper
only swaps the OCRP experiment from the older 4x1-clone/Cartesian checkpoint to
the current direct-Reynolds-isometric IN718 seed-42 checkpoint and writes to a
separate output directory.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "analysis" / "patch_sample_comparison_all_methods_4x4.ipynb"

OLD_EXP = 'repo_root / "experiments" / "IN718" / "iso_embedding_4x4_ocrp_anchorless_4x1clone_01"'
NEW_EXP = (
    'repo_root / "experiments" / "IN718" / "direct_reynolds_isometric_seed_runs" '
    '/ "ocrp_direct_reynolds_isometric_l4_s42"'
)
OLD_OUT = (
    'FIG_ROOT = repo_root / "analysis" / "out" / '
    '"patch_sample_comparison_all_methods_4x4" / sample_id'
)
NEW_OUT = (
    'FIG_ROOT = repo_root / "analysis" / "out" / '
    '"patch_sample_comparison_all_methods_4x4_direct_reynolds" / sample_id'
)


def patch_cell_source(source: str) -> str:
    source = source.replace(
        "SELECTED_CUDA_DEVICE_INDEX = _pick_most_free_gpu()",
        'SELECTED_CUDA_DEVICE_INDEX = int(os.environ.get("NOTEBOOK_CUDA_ORDINAL", "0"))',
    )
    source = source.replace(OLD_EXP, NEW_EXP)
    source = source.replace('checkpoint_name="epoch_0024.pt"', 'checkpoint_name="best_model.pt"')
    source = source.replace("anchorless epoch-24", "direct-Reynolds best")
    source = source.replace("anchorless epoch-24 loaded", "direct-Reynolds OCRP loaded")
    source = source.replace("epoch-24 checkpoint", "direct-Reynolds best checkpoint")
    source = source.replace(OLD_OUT, NEW_OUT)
    return source


def main() -> None:
    os.chdir(REPO_ROOT)
    nb = json.loads(NB_PATH.read_text())
    ns = {"__name__": "__main__"}
    cells_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    for i in cells_to_run:
        print(f"\n===== EXEC notebook cell {i} =====", flush=True)
        t0 = time.time()
        src = patch_cell_source("".join(nb["cells"][i].get("source", [])))
        try:
            exec(compile(src, f"{NB_PATH}:cell{i}:direct_reynolds", "exec"), ns)
        except Exception:
            print(f"ERROR in notebook cell {i}", flush=True)
            traceback.print_exc()
            raise
        print(f"===== DONE cell {i} in {time.time() - t0:.1f}s =====", flush=True)

    print("\nFIG_ROOT", ns.get("FIG_ROOT"))
    print("OCRP checkpoint", ns.get("checkpoint_new_anchorless_ocrp"))
    print("METHODS", list(ns.get("METHODS", {}).keys()))
    print("RESULTS", [k for k in ns.get("results", {}) if not str(k).startswith("__")])


if __name__ == "__main__":
    main()
