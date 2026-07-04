#!/usr/bin/env python3
"""Re-score ALL current Ti_Al_1pct learned-baseline inferences with proper D6 symmetry."""
from __future__ import annotations
import json, sys, os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"); os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
import numpy as np, torch
from scipy.ndimage import binary_dilation
ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "Paper/EBSD_SR_Nature_v3/evals"
for p in (ROOT, EVAL):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import evaluate_new_learned_baselines as enb
import export_test_psnr_ssim_ipf as ipf_eval
from training.train_jangid_baseline import build_model
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry

E = "experiments/Ti_Al_1pct"
ENTRIES = OrderedDict([
    (("4x1","QEDSR"),         (f"{E}/qedsr_4x1_01/inference/test/summary.json", f"{E}/qedsr_4x1_01")),
    (("4x1","Q-RBSA-adapted"),(f"{E}/qrbsa_adapted_4x1_300ep_01/inference/test/summary.json", f"{E}/qrbsa_adapted_4x1_300ep_01")),
    (("4x1","HAN"),           (f"{E}/han_4x1_300ep_01/inference/test/summary.json", f"{E}/han_4x1_300ep_01")),
    (("4x1","RCAN"),          (f"{E}/rcan_4x1_300ep_01/inference/test/summary.json", f"{E}/rcan_4x1_300ep_01")),
    (("4x1","SAN"),           (f"{E}/san_4x1_300ep_01/inference/test/summary.json", f"{E}/san_4x1_300ep_01")),
    (("4x1","Atindama"),      (f"{E}/atindama_inpainting_4x1_01/inference/test/summary.json", None)),
    (("4x1","OCRP"),          (f"{E}/iso_embedding_4x1_ocrp_anchorless_01/inference/test_epoch_0010/summary.json", None)),
    (("4x4","QEDSR"),         (f"{E}/qedsr_4x4_01/inference/test/summary.json", f"{E}/qedsr_4x4_01")),
    (("4x4","Q-RBSA-adapted"),(f"{E}/qrbsa_adapted_4x4_300ep_01/inference/test/summary.json", f"{E}/qrbsa_adapted_4x4_300ep_01")),
    (("4x4","HAN"),           (f"{E}/han_4x4_300ep_01/inference/test/summary.json", f"{E}/han_4x4_300ep_01")),
    (("4x4","RCAN"),          (f"{E}/rcan_4x4_300ep_01/inference/test/summary.json", f"{E}/rcan_4x4_300ep_01")),
    (("4x4","SAN"),           (f"{E}/san_4x4_300ep_01/inference/test/summary.json", f"{E}/san_4x4_300ep_01")),
    (("4x4","Atindama"),      (f"{E}/atindama_inpainting_4x4_01/inference/test/summary.json", None)),
    (("4x4","OCRP"),          (f"{E}/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0044/summary.json", None)),
])
OCRP_P = {"4x1":56641,"4x4":57025}

sym = resolve_symmetry("D6h"); dev = torch.device("cuda")
ops = torch.as_tensor(proper_symmetry_quaternions(sym),dtype=torch.float64,device=dev).clone(); ops[:,1:]*=-1
ipf_eval.SYM = sym
assert proper_symmetry_quaternions(sym).shape[0]==12, "expected 12 proper D6 rotations"

def params(method, exp, task):
    if method=="OCRP": return OCRP_P[task]
    if method=="Atindama":
        from training.train_atindama_inpainting import load_authors_model
        return int(sum(p.numel() for p in load_authors_model(torch.device("cpu")).parameters() if p.requires_grad))
    cfg=json.loads((ROOT/exp/"config.json").read_text())
    return int(sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad))

def orient(path):
    s=json.loads(Path(path).read_text()); mm=[];inter=[];band=[];tp=fp=fn=0;nf=0
    for rec in s["records"]:
        hr=enb._load_record_array(rec,"hr_npy");sr=enb._load_record_array(rec,"sr_npy")
        srt=torch.from_numpy(np.asarray(sr,np.float32)).to(dev,torch.float64);hrt=torch.from_numpy(hr).to(dev,torch.float64)
        mis=enb._misorientation_torch(srt,hrt,ops).cpu().numpy().astype(np.float32)
        pb=enb._boundary_mask_torch(srt,ops).cpu().numpy();rb=enb._boundary_mask_torch(hrt,ops).cpu().numpy()
        rband=binary_dilation(rb,iterations=5)
        tp+=int((pb&rb).sum());fp+=int((pb&~rb).sum());fn+=int((~pb&rb).sum())
        fin=np.isfinite(mis);nf+=int((~fin).sum())
        mm.append(mis[fin]);inter.append(mis[~rband&fin]);band.append(mis[rband&fin])
    mm=np.concatenate(mm);inter=np.concatenate(inter);band=np.concatenate(band)
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return dict(n=s["num_samples"],mean=mm.mean(),med=np.median(mm),p90=np.percentile(mm,90),
                f1=f1,inter=inter.mean(),band=band.mean(),nf=nf)

rows={}
for (task,method),(summ,exp) in ENTRIES.items():
    if not Path(summ).exists(): print("MISSING",task,method,summ); continue
    o=orient(summ); o["p"]=params(method,exp,task); o["summ"]=summ; rows[(task,method)]=o
    print(f"orient {task} {method}: mean={o['mean']:.3f}",flush=True)

def ipf(key):
    s=json.loads(Path(rows[key]["summ"]).read_text()); s["task"]="t"
    r,_=ipf_eval.evaluate_method(s,"m",ipf_eval.provider_from_saved_sr)
    return key,r["psnr_mean_xyz"],r["ssim_mean_xyz"]
with ThreadPoolExecutor(max_workers=8) as pool:
    for fut in as_completed([pool.submit(ipf,k) for k in rows]):
        k,ps,ss=fut.result(); rows[k]["psnr"]=ps; rows[k]["ssim"]=ss

import csv
OUT=ROOT/"analysis/out/ti_all_baselines_corrected.csv"
order=["OCRP","SAN","HAN","RCAN","QEDSR","Q-RBSA-adapted","Atindama"]
with OUT.open("w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["task","method","params","mean_deg","median_deg","p90_deg","boundary_f1","interior_deg","boundary_band_deg","psnr_ipf","ssim_ipf","nonfinite"])
    for task in ("4x1","4x4"):
        print(f"\n===== Ti-7Al-1% {task} (D6 proper, current inferences) =====")
        print(f"{'method':>16}{'params':>11}{'mean':>8}{'median':>8}{'p90':>8}{'bF1':>7}{'interior':>9}{'bBand':>8}{'PSNR':>7}{'SSIM':>8}")
        sub=sorted([(m,rows[(t,m)]) for (t,m) in rows if t==task], key=lambda x:x[1]['mean'])
        for m,r in sub:
            print(f"{m:>16}{r['p']:>11,}{r['mean']:>8.3f}{r['med']:>8.3f}{r['p90']:>8.3f}{r['f1']:>7.3f}{r['inter']:>9.3f}{r['band']:>8.3f}{r['psnr']:>7.2f}{r['ssim']:>8.4f}")
            w.writerow([task,m,r['p'],round(r['mean'],4),round(r['med'],4),round(r['p90'],4),round(r['f1'],4),round(r['inter'],4),round(r['band'],4),round(r['psnr'],4),round(r['ssim'],4),r['nf']])
print(f"\nWrote {OUT}")
