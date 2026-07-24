#!/usr/bin/env python3
"""Render IPF-Z comparison of all methods for the Ti64 zero-shot experiment."""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.symmetry_utils import resolve_symmetry
from utils.quat_ops import format_quaternions
from visualization.ipf_render import render_ipf_rgb, add_ipf_key_panel

ROOT = "/data/home/umang/Materials/Reynolds-QSR_paper/experiments/Zero_shot_performance_Ti64_DIC_Mclean"
OUT = os.path.join(ROOT, "ipf_comparisons")
os.makedirs(OUT, exist_ok=True)

SYM = resolve_symmetry("D6h")

# method label -> sr_quaternions dir (ordered as in Table 3, learned block + OCRP)
LB = os.path.join(ROOT, "learned_baselines_4x4")
METHODS = [
    ("Atindama",  os.path.join(LB, "atindama_inpainting", "sr_quaternions")),
    ("Q-RBSA",    os.path.join(LB, "qrbsaadapted",        "sr_quaternions")),
    ("QEDSR",     os.path.join(LB, "qedsr",               "sr_quaternions")),
    ("RCAN",      os.path.join(LB, "rcan",                "sr_quaternions")),
    ("SAN",       os.path.join(LB, "san",                 "sr_quaternions")),
    ("HAN",       os.path.join(LB, "han",                 "sr_quaternions")),
    ("OCRP (ours)", os.path.join(ROOT, "ocrp_direct_reynolds_isometric_l6_s42",
                                 "inference", "test_best", "sr_quaternions")),
]

def ipfz(q):
    q = format_quaternions(q, normalize=True, hemisphere=True, reduce_fz=True,
                           sym=SYM, to_quat_first=False)
    return render_ipf_rgb(q, SYM, ref_dir="Z")

def nn_up(lr, factor=4):
    return np.repeat(np.repeat(lr, factor, axis=0), factor, axis=1)

def render_sample(sid):
    tag = f"sample_{sid:06d}"
    # LR/HR from OCRP dir (identical across methods)
    ref_dir = METHODS[-1][1]
    lr = np.load(os.path.join(ref_dir, f"{tag}_lr.npy"))
    hr = np.load(os.path.join(ref_dir, f"{tag}_hr.npy"))
    panels = [("LR (input)", ipfz(nn_up(lr)))]
    for name, d in METHODS:
        sr = np.load(os.path.join(d, f"{tag}_sr.npy"))
        panels.append((name, ipfz(sr)))
    panels.append(("HR (target)", ipfz(hr)))

    n = len(panels)
    fig = plt.figure(figsize=(2.05 * n + 2.4, 2.7))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.9], wspace=0.06)
    for i, (name, rgb) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rgb); ax.set_aspect("equal"); ax.axis("off")
        fw = "bold" if name.startswith("OCRP") else "normal"
        ax.set_title(name, fontsize=9, fontweight=fw)
    add_ipf_key_panel(fig, gs[0, -1], SYM, title="IPF-Z key",
                      title_fontsize=8.5, label_fontsize=7.5)
    fig.suptitle(f"Ti64 zero-shot ($4\\times4$, HCP $D_6$) — IPF-Z, {tag}",
                 fontsize=10, y=1.02)
    out = os.path.join(OUT, f"{tag}_ipfz_all_methods.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", out)
    return out

if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [0]
    for s in ids:
        render_sample(s)
