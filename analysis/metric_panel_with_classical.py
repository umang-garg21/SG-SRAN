#!/usr/bin/env python3
"""Metric panel incl. non-learnable baselines (Nearest, Bicubic) and a
distance-to-NN-upsampling diagnostic (tests whether a learned method is just
reproducing nearest-neighbour LR upsampling)."""
import os, sys, glob
import numpy as np
from scipy.ndimage import zoom
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from metric_panel_zeroshot import (mis_deg, tol_mis_deg, boundary_mask, dilate,
                                    tolerant_bf1, grains, basal_pf_hist, norm)
from scipy.stats import wasserstein_distance

def nn_up(lr, f=4):
    return np.repeat(np.repeat(lr, f, 0), f, 1)

def bicubic_up(lr, f=4):
    out = zoom(lr, (f, f, 1), order=3, mode="nearest")
    return norm(out).astype(np.float32)

def metrics_one(sr, hr, nnref):
    m = mis_deg(sr, hr)
    hb = boundary_mask(hr); band = dilate(hb)
    ng_sr, c_sr = grains(sr); ng_hr, c_hr = grains(hr)
    return dict(
        strict=m.mean(),
        tol1=tol_mis_deg(sr, hr, 1).mean(),
        interior=m[~band].mean(),
        boundary=(m[band].mean() if band.any() else np.nan),
        bf1=tolerant_bf1(boundary_mask(sr), hb),
        gratio=ng_sr / max(ng_hr, 1),
        gwass=wasserstein_distance(np.log10(c_sr), np.log10(c_hr)),
        pf=np.abs(basal_pf_hist(sr) - basal_pf_hist(hr)).sum(),
        dNN=mis_deg(sr, nnref).mean(),      # <-- distance to NN-upsampled LR
    )

def run(ref_dir, method_dirs, n, title):
    cols = ["strict","tol1","interior","boundary","bf1","gratio","gwass","pf","dNN"]
    agg = {name: {c: [] for c in cols} for name, _ in method_dirs}
    agg["Nearest"] = {c: [] for c in cols}
    agg["Bicubic"] = {c: [] for c in cols}
    order = ["Nearest", "Bicubic"] + [n for n, _ in method_dirs]
    for i in range(n):
        tag = f"sample_{i:06d}"
        lr = np.load(f"{ref_dir}/{tag}_lr.npy").astype(np.float32)
        hr = np.load(f"{ref_dir}/{tag}_hr.npy").astype(np.float32)
        nnref = nn_up(lr)
        for name, sr in [("Nearest", nnref), ("Bicubic", bicubic_up(lr))]:
            for k, v in metrics_one(sr, hr, nnref).items():
                agg[name][k].append(v)
        for name, d in method_dirs:
            sr = np.load(f"{d}/{tag}_sr.npy").astype(np.float32)
            for k, v in metrics_one(sr, hr, nnref).items():
                agg[name][k].append(v)
    lines = [title, f"n={n} patches (HCP D6). dNN = mean misorient to NN-upsampled LR (small => behaves like NN).", ""]
    lines.append(f"{'method':10s} " + " ".join(f"{c:>9s}" for c in cols))
    for name in order:
        r = {c: float(np.nanmean(agg[name][c])) for c in cols}
        lines.append(f"{name:10s} " + " ".join(f"{r[c]:9.3f}" for c in cols))
    return "\n".join(lines)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ti64"
    if which == "ti64":
        R = "experiments/Zero_shot_performance_Ti64_DIC_Mclean"
        LB = f"{R}/learned_baselines_4x4"
        ref = f"{R}/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions"
        md = [("Atindama",f"{LB}/atindama_inpainting/sr_quaternions"),
              ("Q-RBSA",f"{LB}/qrbsaadapted/sr_quaternions"),
              ("QEDSR",f"{LB}/qedsr/sr_quaternions"),
              ("RCAN",f"{LB}/rcan/sr_quaternions"),
              ("SAN",f"{LB}/san/sr_quaternions"),
              ("HAN",f"{LB}/han/sr_quaternions"),
              ("OCRP",ref)]
        title = "Ti64 zero-shot (OOD)"
    else:
        base = "experiments/Ti_Al_1pct"
        ref = f"{base}/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions"
        md = [("Atindama",f"{base}/atindama_inpainting_4x4_01/inference/test/sr_quaternions"),
              ("Q-RBSA",f"{base}/qrbsa_adapted_4x4_300ep_01/inference/test/sr_quaternions"),
              ("QEDSR",f"{base}/qedsr_4x4_01/inference/test/sr_quaternions"),
              ("RCAN",f"{base}/rcan_4x4_300ep_01/inference/test/sr_quaternions"),
              ("SAN",f"{base}/san_4x4_300ep_01/inference/test_epoch0205/sr_quaternions"),
              ("HAN",f"{base}/han_4x4_300ep_01/inference/test/sr_quaternions"),
              ("OCRP",ref)]
        title = "In-distribution Ti-6Al-4V test"
    n = len(glob.glob(f"{ref}/*_sr.npy"))
    out = run(ref, md, n, title)
    print(out)
    dst = f"analysis/out/metric_panel_{which}.txt"
    open(dst, "w").write(out);
