#!/usr/bin/env python3
"""Full IN718 learned-baseline table, Oh symmetry, best checkpoints (OCRP 4x1=ep30, 4x4=ep25)."""
from __future__ import annotations
import json, sys, os, csv
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

E = "experiments/IN718"
ENT = OrderedDict([
  (("4x1","EDSR"),   (f"{E}/edsr_4x1_01/inference/test/summary.json", f"{E}/edsr_4x1_01","best_model.pt")),
  (("4x1","QEDSR"),  (f"{E}/qedsr_4x1_01/inference/test/summary.json", f"{E}/qedsr_4x1_01","best_model.pt")),
  (("4x1","Q-RBSA-adapted"),(f"{E}/qrbsa_4x1_300ep_01/inference/test/summary.json", f"{E}/qrbsa_4x1_300ep_01","best_model.pt")),
  (("4x1","HAN"),    (f"{E}/han_4x1_300ep_01/inference/test_best/summary.json", f"{E}/han_4x1_300ep_01","best_model.pt")),
  (("4x1","RCAN"),   (f"{E}/rcan_4x1_300ep_01/inference/test_best/summary.json", f"{E}/rcan_4x1_300ep_01","best_model.pt")),
  (("4x1","SAN"),    (f"{E}/san_4x1_300ep_01/inference/test/summary.json", f"{E}/san_4x1_300ep_01","best_model.pt")),
  (("4x1","OCRP"),   (f"{E}/iso_embedding_4x1_ocrp_anchorless_01/inference/test_epoch_0030/summary.json", None,"epoch_0030")),
  (("4x4","EDSR"),   (f"{E}/edsr_4x4_01/inference/test/summary.json", f"{E}/edsr_4x4_01","best_model.pt")),
  (("4x4","QEDSR"),  (f"{E}/qedsr_4x4_01/inference/test/summary.json", f"{E}/qedsr_4x4_01","best_model.pt")),
  (("4x4","Q-RBSA-adapted"),(f"{E}/qrbsa_4x4_300ep_01/inference/test/summary.json", f"{E}/qrbsa_4x4_300ep_01","best_model.pt")),
  (("4x4","HAN"),    (f"{E}/han_4x4_300ep_01/inference/test/summary.json", f"{E}/han_4x4_300ep_01","best_model.pt")),
  (("4x4","RCAN"),   (f"{E}/rcan_4x4_300ep_01/inference/test/summary.json", f"{E}/rcan_4x4_300ep_01","best_model.pt")),
  (("4x4","SAN"),    (f"{E}/san_4x4_300ep_01/inference/test/summary.json", f"{E}/san_4x4_300ep_01","best_model.pt")),
  (("4x4","OCRP"),   (f"{E}/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0024/summary.json", None,"epoch_0024")),
])
OCRP_P = {"4x1":56641,"4x4":57025}
sym = resolve_symmetry("Oh"); dev = torch.device("cuda")
ops = torch.as_tensor(proper_symmetry_quaternions(sym),dtype=torch.float64,device=dev).clone(); ops[:,1:]*=-1
ipf_eval.SYM = sym

def params(method, exp, task):
    if method=="OCRP": return OCRP_P[task]
    cfg=json.loads((ROOT/exp/"config.json").read_text())
    return int(sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad))

def orient(path):
    s=json.loads(Path(path).read_text());mm=[];inter=[];band=[];tp=fp=fn=0;nf=0
    for rec in s["records"]:
        hr=enb._load_record_array(rec,"hr_npy");sr=enb._load_record_array(rec,"sr_npy")
        srt=torch.from_numpy(np.asarray(sr,np.float32)).to(dev,torch.float64);hrt=torch.from_numpy(hr).to(dev,torch.float64)
        mis=enb._misorientation_torch(srt,hrt,ops).cpu().numpy().astype(np.float32)
        pb=enb._boundary_mask_torch(srt,ops).cpu().numpy();rb=enb._boundary_mask_torch(hrt,ops).cpu().numpy()
        rband=binary_dilation(rb,iterations=5)
        tp+=int((pb&rb).sum());fp+=int((pb&~rb).sum());fn+=int((~pb&rb).sum())
        fin=np.isfinite(mis);nf+=int((~fin).sum())
        mm.append(mis[fin]);inter.append(mis[~rband&fin]);band.append(mis[rband&fin])
    mm=np.concatenate(mm);f1=2*tp/(2*tp+fp+fn)
    return dict(n=s["num_samples"],mean=mm.mean(),med=np.median(mm),p90=np.percentile(mm,90),f1=f1,
                inter=np.concatenate(inter).mean(),band=np.concatenate(band).mean(),nf=nf)

rows={}
for (task,method),(summ,exp,ck) in ENT.items():
    if not Path(summ).exists(): print("MISSING",task,method,summ,flush=True); continue
    o=orient(summ);o["p"]=params(method,exp,task);o["summ"]=summ;o["ck"]=ck;rows[(task,method)]=o
    print(f"orient {task} {method}: mean={o['mean']:.3f}",flush=True)

def ipf(key):
    s=json.loads(Path(rows[key]["summ"]).read_text());s["task"]="t"
    r,_=ipf_eval.evaluate_method(s,"m",ipf_eval.provider_from_saved_sr)
    return key,r["psnr_mean_xyz"],r["ssim_mean_xyz"]
with ThreadPoolExecutor(max_workers=8) as pool:
    for fut in as_completed([pool.submit(ipf,k) for k in rows]):
        k,ps,ss=fut.result();rows[k]["psnr"]=ps;rows[k]["ssim"]=ss

OUT=ROOT/"analysis/out/in718_all_baselines_corrected.csv"
with OUT.open("w",newline="") as fh:
    w=csv.writer(fh);w.writerow(["task","method","checkpoint","params","n_test","mean_deg","median_deg","p90_deg","boundary_f1","interior_deg","boundary_band_deg","psnr_ipf","ssim_ipf","nonfinite"])
    for task in ("4x1","4x4"):
        print(f"\n===== IN718 {task} (Oh; best checkpoints) n=147 =====")
        print(f"{'method':>16}{'params':>11}{'mean':>8}{'median':>8}{'p90':>8}{'bF1':>7}{'interior':>9}{'bBand':>8}{'PSNR':>7}{'SSIM':>8}")
        for m,r in sorted([(m,rows[(t,m)]) for (t,m) in rows if t==task],key=lambda x:x[1]['mean']):
            print(f"{m:>16}{r['p']:>11,}{r['mean']:>8.3f}{r['med']:>8.3f}{r['p90']:>8.3f}{r['f1']:>7.3f}{r['inter']:>9.3f}{r['band']:>8.3f}{r['psnr']:>7.2f}{r['ssim']:>8.4f}")
            w.writerow([task,m,r['ck'],r['p'],r['n'],round(r['mean'],4),round(r['med'],4),round(r['p90'],4),round(r['f1'],4),round(r['inter'],4),round(r['band'],4),round(r['psnr'],4),round(r['ssim'],4),r['nf']])
print(f"\nWrote {OUT}")
