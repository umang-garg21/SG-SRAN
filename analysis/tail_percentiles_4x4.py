#!/usr/bin/env python3
"""Tail percentiles (p90/p95/p99) of misorientation for all 8 methods, both
materials, mean+/-std over 5 seeds. OCRP from the all-epochs anchorless runs;
baselines from the seed study. Orientation-only (no PSNR/SSIM) for speed.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib"); os.environ.setdefault("NUMBA_CACHE_DIR","/tmp/numba")
import numpy as np, torch
ROOT=Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT/"Paper/EBSD_SR_Nature_v3/evals"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import evaluate_new_learned_baselines as enb
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry

SEEDS=[42,43,44,45,46]
MODELS=["edsr","qedsr","qrbsa","han","rcan","san","ocrp","atindama"]
LABEL={"edsr":"EDSR","qedsr":"QEDSR","qrbsa":"Q-RBSA","han":"HAN","rcan":"RCAN","san":"SAN","ocrp":"OCRP","atindama":"Atindama"}
MATS=[("IN718","Oh","in718"),("Ti_Al_1pct","D6h","ti")]
PCT=[90,95,99]

def summ_path(mat,m,s):
    if m=="ocrp":
        return ROOT/f"experiments/{mat}/iso_embedding_4x4_ocrp_anchorless_allepochs_s{s}/inference/test/summary.json"
    return ROOT/f"experiments/{mat}/seed_runs/{m}_4x4_s{s}/inference/test/summary.json"

dev=torch.device("cuda")
def tails(summ,ops):
    s=json.loads(Path(summ).read_text()); allmis=[]
    for rec in s["records"]:
        hr=enb._load_record_array(rec,"hr_npy");sr=enb._load_record_array(rec,"sr_npy")
        srt=torch.from_numpy(np.asarray(sr,np.float32)).to(dev,torch.float64);hrt=torch.from_numpy(hr).to(dev,torch.float64)
        mis=enb._misorientation_torch(srt,hrt,ops).cpu().numpy()
        allmis.append(mis[np.isfinite(mis)])
    a=np.concatenate(allmis)
    return {"mean":float(a.mean()),"median":float(np.median(a)),**{f"p{p}":float(np.percentile(a,p)) for p in PCT}}

cols=["mean","median"]+[f"p{p}" for p in PCT]
for mat,symname,tag in MATS:
    sym=resolve_symmetry(symname)
    ops=torch.as_tensor(proper_symmetry_quaternions(sym),dtype=torch.float64,device=dev).clone(); ops[:,1:]*=-1
    by=defaultdict(list)
    for m in MODELS:
        for s in SEEDS:
            p=summ_path(mat,m,s)
            if not p.exists(): print("MISSING",mat,m,s); continue
            by[m].append(tails(str(p),ops))
    out=ROOT/f"analysis/out/tail_pct_{tag}.csv"
    rows=[]
    with out.open("w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["model","n_seeds"]+[f"{c}_mean" for c in cols]+[f"{c}_std" for c in cols])
        for m in MODELS:
            rs=by[m]; mu={c:float(np.mean([r[c] for r in rs])) for c in cols}; sd={c:float(np.std([r[c] for r in rs],ddof=1)) for c in cols}
            w.writerow([LABEL[m],len(rs)]+[round(mu[c],4) for c in cols]+[round(sd[c],4) for c in cols]); rows.append((m,mu,sd))
    print(f"\n===== {mat} 4x4 ({symname}) misorientation tails, mean+/-std over 5 seeds =====")
    print(f"{'model':>10} {'mean':>11} {'median':>11} {'p90':>11} {'p95':>11} {'p99':>11}")
    for m,mu,sd in sorted(rows,key=lambda x:x[1]['p99']):
        f=lambda k:f"{mu[k]:.2f}±{sd[k]:.2f}"
        print(f"{LABEL[m]:>10} {f('mean'):>11} {f('median'):>11} {f('p90'):>11} {f('p95'):>11} {f('p99'):>11}")
    print(f"Wrote {out.name}")
print("\nEXIT_OK")
