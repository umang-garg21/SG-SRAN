#!/usr/bin/env python3
"""Final 4x4 comparison tables (IN718 + Ti_Al_1pct), mean+/-std over 5 seeds.

Baselines come from the seed study (experiments/<mat>/seed_runs); OCRP comes from
the all-epochs anchorless runs (iso_embedding_4x4_ocrp_anchorless_allepochs_s*),
both evaluated at best_model.pt with one shared eval pipeline.
"""
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

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["edsr", "qedsr", "qrbsa", "han", "rcan", "san", "ocrp", "atindama"]
LABEL = {"edsr":"EDSR","qedsr":"QEDSR","qrbsa":"Q-RBSA","han":"HAN","rcan":"RCAN","san":"SAN","ocrp":"OCRP","atindama":"Atindama"}
MATERIALS = [("IN718", "Oh"), ("Ti_Al_1pct", "D6h")]

def summ_path(mat, m, s):
    if m == "ocrp":
        return ROOT/f"experiments/{mat}/iso_embedding_4x4_ocrp_anchorless_allepochs_s{s}/inference/test/summary.json"
    return ROOT/f"experiments/{mat}/seed_runs/{m}_4x4_s{s}/inference/test/summary.json"

def orient(path, ops, dev):
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

def ipf(rec):
    s=json.loads(Path(rec["summ"]).read_text());s["task"]="t"
    r,_=ipf_eval.evaluate_method(s,"m",ipf_eval.provider_from_saved_sr)
    return id(rec),r["psnr_mean_xyz"],r["ssim_mean_xyz"]

metrics=["mean","median","p90","bf1","interior","bband","psnr","ssim"]
dev=torch.device("cuda")

for mat, symname in MATERIALS:
    sym=resolve_symmetry(symname); ipf_eval.SYM=sym
    ops=torch.as_tensor(proper_symmetry_quaternions(sym),dtype=torch.float64,device=dev).clone(); ops[:,1:]*=-1
    per_seed=[]
    for m in MODELS:
        for s in SEEDS:
            p=summ_path(mat,m,s)
            if not p.exists(): print("MISSING",mat,m,s,p); continue
            o=orient(str(p),ops,dev); o["model"]=m; o["seed"]=s; o["summ"]=str(p); per_seed.append(o)
    idx={id(r):r for r in per_seed}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(ipf,r) for r in per_seed]):
            k,ps,ss=fut.result(); idx[k]["psnr"]=ps; idx[k]["ssim"]=ss

    tag=mat.lower().replace("_al_1pct","").replace("in718","in718")
    PS=ROOT/f"analysis/out/final_4x4_{tag}_per_seed.csv"
    with PS.open("w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["model","seed","n"]+metrics+["nonfin"])
        for r in per_seed: w.writerow([LABEL[r["model"]],r["seed"],r["n"]]+[round(r[k],4) for k in metrics]+[r["nonfin"]])
    by=defaultdict(list)
    for r in per_seed: by[r["model"]].append(r)
    AGG=ROOT/f"analysis/out/final_4x4_{tag}_summary.csv"
    rows=[]
    with AGG.open("w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["model","params_M","n_seeds"]+[f"{x}_mean" for x in metrics]+[f"{x}_std" for x in metrics])
        for m in MODELS:
            rs=by[m]
            mu={x:float(np.mean([r[x] for r in rs])) for x in metrics}; sd={x:float(np.std([r[x] for r in rs],ddof=1)) for x in metrics}
            w.writerow([LABEL[m],"",len(rs)]+[round(mu[x],4) for x in metrics]+[round(sd[x],4) for x in metrics])
            rows.append((m,mu,sd))
    print(f"\n===== FINAL {mat} 4x4 over {len(SEEDS)} seeds (mean +/- std) =====")
    hdr=f"{'model':>10} {'mean':>12} {'median':>12} {'p90':>12} {'bF1':>13} {'interior':>12} {'bBand':>12} {'PSNR':>12} {'SSIM':>15}"
    print(hdr)
    for m,mu,sd in sorted(rows,key=lambda x:x[1]['mean']):
        f=lambda k,p=2:f"{mu[k]:.{p}f}±{sd[k]:.{p}f}"
        print(f"{LABEL[m]:>10} {f('mean'):>12} {f('median'):>12} {f('p90'):>12} {f('bf1',3):>13} {f('interior'):>12} {f('bband'):>12} {f('psnr'):>12} {f('ssim',4):>15}")
    print(f"Wrote {PS.name}, {AGG.name}")
