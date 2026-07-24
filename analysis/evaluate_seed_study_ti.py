#!/usr/bin/env python3
"""Aggregate the IN718 4x4 5-seed study: per-seed metrics + mean/std per model."""
from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
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
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry

SR = "experiments/Ti_Al_1pct/seed_runs"
MODELS = ["edsr", "qedsr", "qrbsa", "han", "rcan", "san", "ocrp", "atindama"]
SEEDS = [42, 43, 44, 45, 46]
LABEL = {"edsr":"EDSR","qedsr":"QEDSR","qrbsa":"Q-RBSA","han":"HAN","rcan":"RCAN","san":"SAN","ocrp":"OCRP","atindama":"Atindama"}

sym = resolve_symmetry("D6h"); dev = torch.device("cuda")
ops = torch.as_tensor(proper_symmetry_quaternions(sym), dtype=torch.float64, device=dev).clone(); ops[:, 1:] *= -1
ipf_eval.SYM = sym

def orient(path):
    s = json.loads(Path(path).read_text()); mm=[];inter=[];band=[];tp=fp=fn=0;nf=0
    for rec in s["records"]:
        hr=enb._load_record_array(rec,"hr_npy");sr=enb._load_record_array(rec,"sr_npy")
        srt=torch.from_numpy(np.asarray(sr,np.float32)).to(dev,torch.float64);hrt=torch.from_numpy(hr).to(dev,torch.float64)
        mis=enb._misorientation_torch(srt,hrt,ops).cpu().numpy().astype(np.float32)
        pb=enb._boundary_mask_torch(srt,ops).cpu().numpy();rb=enb._boundary_mask_torch(hrt,ops).cpu().numpy()
        rband=binary_dilation(rb,iterations=5)
        tp+=int((pb&rb).sum());fp+=int((pb&~rb).sum());fn+=int((~pb&rb).sum())
        fin=np.isfinite(mis);nf+=int((~fin).sum())
        mm.append(mis[fin]);inter.append(mis[~rband&fin]);band.append(mis[rband&fin])
    mm=np.concatenate(mm);f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return dict(n=s["num_samples"],mean=float(mm.mean()),median=float(np.median(mm)),p90=float(np.percentile(mm,90)),
                bf1=float(f1),interior=float(np.concatenate(inter).mean()),bband=float(np.concatenate(band).mean()),nonfin=nf)

# collect per-seed
per_seed=[]
for m in MODELS:
    for s in SEEDS:
        summ=ROOT/SR/f"{m}_4x4_s{s}"/"inference/test/summary.json"
        if not summ.exists(): print("MISSING",m,s); continue
        o=orient(str(summ)); o["model"]=m; o["seed"]=s; o["summ"]=str(summ); per_seed.append(o)
        print(f"orient {m} s{s}: mean={o['mean']:.3f}",flush=True)

def ipf(rec):
    s=json.loads(Path(rec["summ"]).read_text());s["task"]="t"
    r,_=ipf_eval.evaluate_method(s,"m",ipf_eval.provider_from_saved_sr)
    return id(rec),r["psnr_mean_xyz"],r["ssim_mean_xyz"]
idx={id(r):r for r in per_seed}
with ThreadPoolExecutor(max_workers=8) as pool:
    for fut in as_completed([pool.submit(ipf,r) for r in per_seed]):
        k,ps,ss=fut.result(); idx[k]["psnr"]=ps; idx[k]["ssim"]=ss

# write per-seed csv
PS=ROOT/"analysis/out/seed_study_ti_4x4_per_seed.csv"
cols=["model","seed","n","mean","median","p90","bf1","interior","bband","psnr","ssim","nonfin"]
with PS.open("w",newline="") as fh:
    w=csv.writer(fh);w.writerow(cols)
    for r in per_seed: w.writerow([LABEL[r["model"]],r["seed"],r["n"]]+[round(r[k],4) for k in ("mean","median","p90","bf1","interior","bband","psnr","ssim")]+[r["nonfin"]])

# aggregate mean+/-std
by=defaultdict(list)
for r in per_seed: by[r["model"]].append(r)
AGG=ROOT/"analysis/out/seed_study_ti_4x4_summary.csv"
metrics=["mean","median","p90","bf1","interior","bband","psnr","ssim"]
with AGG.open("w",newline="") as fh:
    w=csv.writer(fh);w.writerow(["model","n_seeds"]+[f"{x}_mean" for x in metrics]+[f"{x}_std" for x in metrics])
    rows=[]
    for m in MODELS:
        rs=by[m];
        means={x:np.mean([r[x] for r in rs]) for x in metrics}; stds={x:np.std([r[x] for r in rs],ddof=1) for x in metrics}
        w.writerow([LABEL[m],len(rs)]+[round(means[x],4) for x in metrics]+[round(stds[x],4) for x in metrics])
        rows.append((m,len(rs),means,stds))

print("\n===== Ti_Al_1pct 4x4 over 5 seeds (mean +/- std) =====")
print(f"{'model':>12}{'mean':>16}{'median':>15}{'p90':>15}{'bF1':>15}{'PSNR':>15}{'SSIM':>16}")
for m,ns,mu,sd in sorted(rows,key=lambda x:x[2]['mean']):
    f=lambda k,p=2:f"{mu[k]:.{p}f}+/-{sd[k]:.{p}f}"
    print(f"{LABEL[m]:>12}{f('mean'):>16}{f('median'):>15}{f('p90'):>15}{f('bf1',3):>15}{f('psnr'):>15}{f('ssim',4):>16}")
print(f"\nWrote {PS}\nWrote {AGG}")
