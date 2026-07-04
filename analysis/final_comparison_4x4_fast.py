#!/usr/bin/env python3
"""Final 4x4 comparison tables. Reuse the 7 baseline rows already computed in the
seed-study per-seed CSVs (identical eval pipeline + identical inference summaries);
recompute only OCRP from the all-epochs anchorless best_model.pt inferences.
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
metrics = ["mean","median","p90","bf1","interior","bband","psnr","ssim"]
# label -> approx trainable params (millions), for the final table
PARAMS = {"EDSR":6.36,"QEDSR":1.82,"Q-RBSA":6.17,"HAN":16.07,"RCAN":15.59,"SAN":15.94,"OCRP":0.049,"Atindama":25.8}
ORDER = ["OCRP","SAN","RCAN","HAN","EDSR","QEDSR","Q-RBSA","Atindama"]  # display only; re-sorted by mean

MATS = [
    ("IN718", "Oh",  "analysis/out/seed_study_in718_4x4_per_seed.csv", "in718"),
    ("Ti_Al_1pct", "D6h", "analysis/out/seed_study_ti_4x4_per_seed.csv", "ti"),
]

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

def ipf_eval_one(summ):
    s=json.loads(Path(summ).read_text());s["task"]="t"
    r,_=ipf_eval.evaluate_method(s,"m",ipf_eval.provider_from_saved_sr)
    return r["psnr_mean_xyz"],r["ssim_mean_xyz"]

dev=torch.device("cuda")
for mat, symname, csvpath, tag in MATS:
    sym=resolve_symmetry(symname); ipf_eval.SYM=sym
    ops=torch.as_tensor(proper_symmetry_quaternions(sym),dtype=torch.float64,device=dev).clone(); ops[:,1:]*=-1

    # 1) reuse baseline rows (everything except OCRP) from the seed-study per-seed CSV
    rows=[]
    for r in csv.DictReader(open(ROOT/csvpath)):
        lab=r["model"].replace("(lr1e-4)","")
        if lab=="OCRP": continue
        rows.append({"model":lab,"seed":int(r["seed"]),**{k:float(r[k]) for k in metrics}})

    # 2) recompute OCRP fresh from the all-epochs anchorless best_model.pt inferences
    ocrp=[]
    for s in SEEDS:
        summ=ROOT/f"experiments/{mat}/iso_embedding_4x4_ocrp_anchorless_allepochs_s{s}/inference/test/summary.json"
        o=orient(str(summ),ops,dev); o["summ"]=str(summ); o["seed"]=s; ocrp.append(o)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs={pool.submit(ipf_eval_one,o["summ"]):o for o in ocrp}
        for fut in as_completed(futs):
            ps,ss=fut.result(); futs[fut]["psnr"]=ps; futs[fut]["ssim"]=ss
    for o in ocrp:
        rows.append({"model":"OCRP","seed":o["seed"],**{k:o[k] for k in metrics}})

    # 3) aggregate mean+/-std per model
    by=defaultdict(list)
    for r in rows: by[r["model"]].append(r)
    PS=ROOT/f"analysis/out/final_4x4_{tag}_per_seed.csv"
    with PS.open("w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["model","seed"]+metrics)
        for r in sorted(rows,key=lambda x:(x["model"],x["seed"])):
            w.writerow([r["model"],r["seed"]]+[round(r[k],4) for k in metrics])
    AGG=ROOT/f"analysis/out/final_4x4_{tag}_summary.csv"
    agg=[]
    with AGG.open("w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["model","params_M","n_seeds"]+[f"{x}_mean" for x in metrics]+[f"{x}_std" for x in metrics])
        for m in by:
            rs=by[m]; mu={x:float(np.mean([r[x] for r in rs])) for x in metrics}; sd={x:float(np.std([r[x] for r in rs],ddof=1)) for x in metrics}
            agg.append((m,mu,sd))
        for m,mu,sd in sorted(agg,key=lambda x:x[1]["mean"]):
            w.writerow([m,PARAMS.get(m,""),len(by[m])]+[round(mu[x],4) for x in metrics]+[round(sd[x],4) for x in metrics])

    print(f"\n===== FINAL {mat} 4x4 ({symname}) over 5 seeds (mean +/- std) =====")
    print(f"{'model':>10} {'params':>7} {'mean':>11} {'median':>11} {'p90':>11} {'bF1':>13} {'interior':>11} {'bBand':>11} {'PSNR':>11} {'SSIM':>14}")
    for m,mu,sd in sorted(agg,key=lambda x:x[1]["mean"]):
        f=lambda k,p=2:f"{mu[k]:.{p}f}±{sd[k]:.{p}f}"
        print(f"{m:>10} {PARAMS.get(m,0):>6.3f}M {f('mean'):>11} {f('median'):>11} {f('p90'):>11} {f('bf1',3):>13} {f('interior'):>11} {f('bband'):>11} {f('psnr'):>11} {f('ssim',4):>14}")
    print(f"Wrote {PS.name}, {AGG.name}")
