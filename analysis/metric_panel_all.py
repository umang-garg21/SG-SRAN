#!/usr/bin/env python3
"""Finalized metric panel across ALL datasets/methods, correct symmetry per material.
Columns: mean median p90 p95 (pooled per-pixel) | tol1 interior boundary bf1 gratio gwass dNN (per-sample)."""
import os, sys, glob
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import wasserstein_distance
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.symmetry_utils import proper_symmetry_quaternions

def qmul(a,b):
    aw,ax,ay,az=a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],-1)
def norm(q): return q/np.clip(np.linalg.norm(q,axis=-1,keepdims=True),1e-12,None)

def make_mis(SYM):
    def mis_deg(pred,tgt):
        H,W=pred.shape[:2]; p=norm(pred.reshape(-1,4)); t=norm(tgt.reshape(-1,4))
        eq=qmul(SYM[:,None,:],t[None,:,:]); best=np.clip(np.abs((eq*p[None,:,:]).sum(-1)).max(0),0,1)
        return np.rad2deg(2*np.arccos(best)).reshape(H,W)
    return mis_deg

def dilate(m):
    d=m.copy()
    for dy in(-1,0,1):
        for dx in(-1,0,1): d|=np.roll(np.roll(m,dy,0),dx,1)
    return d

def run_dataset(name, sym_name, ref_dir, method_dirs, f=4):
    SYM=proper_symmetry_quaternions(sym_name).astype(np.float32)
    mis=make_mis(SYM)
    def bmask(q,thr=5.):
        b=np.zeros(q.shape[:2],bool); h=mis(q[:,:-1],q[:,1:])>thr; b[:,:-1]|=h; b[:,1:]|=h
        v=mis(q[:-1],q[1:])>thr; b[:-1]|=v; b[1:]|=v; return b
    def bf1t(pb,rb):
        rd=dilate(rb); pd=dilate(pb)
        prec=(pb&rd).sum()/max(pb.sum(),1); rec=(rb&pd).sum()/max(rb.sum(),1)
        return 2*prec*rec/max(prec+rec,1e-9)
    def grains(q,thr=5.):
        H,W=q.shape[:2]; N=H*W; idx=np.arange(N).reshape(H,W); ii=[];jj=[]
        sh=mis(q[:,:-1],q[:,1:])<=thr; ii.append(idx[:,:-1][sh]); jj.append(idx[:,1:][sh])
        sv=mis(q[:-1],q[1:])<=thr; ii.append(idx[:-1][sv]); jj.append(idx[1:][sv])
        ii=np.concatenate(ii); jj=np.concatenate(jj)
        n,_=connected_components(coo_matrix((np.ones_like(ii),(ii,jj)),shape=(N,N)),directed=False)
        return n
    def gsz(q,thr=5.):
        H,W=q.shape[:2]; N=H*W; idx=np.arange(N).reshape(H,W); ii=[];jj=[]
        sh=mis(q[:,:-1],q[:,1:])<=thr; ii.append(idx[:,:-1][sh]); jj.append(idx[:,1:][sh])
        sv=mis(q[:-1],q[1:])<=thr; ii.append(idx[:-1][sv]); jj.append(idx[1:][sv])
        ii=np.concatenate(ii); jj=np.concatenate(jj)
        _,lab=connected_components(coo_matrix((np.ones_like(ii),(ii,jj)),shape=(N,N)),directed=False)
        return np.unique(lab,return_counts=True)[1]
    def tol1(pred,tgt):
        best=np.full(pred.shape[:2],999.,np.float32)
        for dy in(-1,0,1):
            for dx in(-1,0,1): best=np.minimum(best,mis(pred,np.roll(np.roll(tgt,dy,0),dx,1)))
        return best.mean()

    ref_sr=ref_dir
    n=len(glob.glob(f"{ref_sr}/*_sr.npy"))
    hr0=np.load(f"{ref_sr}/sample_000000_hr.npy")
    # build provider list; Nearest/Bicubic computed from LR
    provs=[("Nearest","nn"),("Bicubic","bic")]+[(nm,d) for nm,d in method_dirs]
    MINSZ=int(os.environ.get("MINSZ","5"))  # min grain size (px): sub-MINSZ components = speckle
    def realsz(a):  # sizes of grains >= MINSZ (log10)
        b=a[a>=MINSZ]; return np.log10(b) if b.size else np.log10(a[:1])
    acc={nm:dict(pix=[],tol=[],ii=[],bb=[],bf=[],gr=[],gw=[],sp=[],dn=[]) for nm,_ in provs}
    skip=set()
    for i in range(n):
        tag=f"sample_{i:06d}"
        lr=np.load(f"{ref_sr}/{tag}_lr.npy").astype(np.float32)
        hr=np.load(f"{ref_sr}/{tag}_hr.npy").astype(np.float32)
        nnref=np.repeat(np.repeat(lr,f,0),f,1)
        hb=bmask(hr); band=dilate(hb)
        hsz=gsz(hr); hr5=max(int((hsz>=MINSZ).sum()),1); szhr=realsz(hsz)
        for nm,src in provs:
            if nm in skip: continue
            if src=="nn": sr=nnref
            elif src=="bic": sr=norm(zoom(lr,(f,f,1),order=3,mode="nearest")).astype(np.float32)
            else:
                p=f"{src}/{tag}_sr.npy"
                if not os.path.exists(p): skip.add(nm); continue
                sr=np.load(p).astype(np.float32)
                if i==0 and not (sr.shape==hr.shape): skip.add(nm); continue
            m=mis(sr,hr); ssz=gsz(sr)
            a=acc[nm]; a["pix"].append(m.ravel()); a["tol"].append(tol1(sr,hr))
            a["ii"].append(m[~band].mean()); a["bb"].append(m[band].mean() if band.any() else np.nan)
            a["bf"].append(bf1t(bmask(sr),hb))
            a["gr"].append((ssz>=MINSZ).sum()/hr5)                      # >=5px grain-count ratio
            a["gw"].append(wasserstein_distance(realsz(ssz),szhr))       # >=5px grain-size Wasserstein
            a["sp"].append(100.0*ssz[ssz<MINSZ].sum()/ssz.sum())         # speckle %: pixels in <5px specks
            a["dn"].append(mis(sr,nnref).mean())
    cols=["mean","median","p90","p95","tol1","interior","boundary","bf1",f"gr{MINSZ}",f"gwass{MINSZ}","speck%","dNN"]
    out=[f"### {name}  (symmetry {sym_name}, n={n})",
         f"{'method':10s} "+" ".join(f"{c:>8s}" for c in cols)]
    order=[nm for nm,_ in provs if nm not in skip]
    rows={}
    for nm in order:
        a=acc[nm]
        if not a["pix"]: continue
        pix=np.concatenate(a["pix"])
        r={"mean":pix.mean(),"median":np.median(pix),"p90":np.percentile(pix,90),"p95":np.percentile(pix,95),
           "tol1":np.nanmean(a["tol"]),"interior":np.nanmean(a["ii"]),"boundary":np.nanmean(a["bb"]),
           "bf1":np.nanmean(a["bf"]),f"gr{MINSZ}":np.nanmean(a["gr"]),f"gwass{MINSZ}":np.nanmean(a["gw"]),
           "speck%":np.nanmean(a["sp"]),"dNN":np.nanmean(a["dn"])}
        rows[nm]=r
        out.append(f"{nm:10s} "+" ".join(f"{r[c]:8.3f}" for c in cols))
    return "\n".join(out)

REG={
 "IN718_indist":("Oh","experiments/IN718/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l4_s42/inference/test_best/sr_quaternions",
   [("Atindama","experiments/IN718/atindama_inpainting_4x4_01/inference/test/sr_quaternions"),
    ("Q-RBSA","experiments/IN718/qrbsa_4x4_300ep_01/inference/test/sr_quaternions"),
    ("QEDSR","experiments/IN718/qedsr_4x4_01/inference/test/sr_quaternions"),
    ("RCAN","experiments/IN718/rcan_4x4_300ep_01/inference/test_ep171/sr_quaternions"),
    ("SAN","experiments/IN718/san_4x4_300ep_01/inference/test_ep199/sr_quaternions"),
    ("HAN","experiments/IN718/han_4x4_300ep_01/inference/test/sr_quaternions"),
    ("OCRP","experiments/IN718/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l4_s42/inference/test_best/sr_quaternions")]),
 "CoNi_zeroshot":("Oh","experiments/Zero_shot_performance_CoNi_x250/ocrp_direct_reynolds_isometric_l4_s42/inference/train_best/sr_quaternions",
   [("Atindama","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
    ("Q-RBSA","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
    ("QEDSR","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/qedsr/sr_quaternions"),
    ("RCAN","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/rcan/sr_quaternions"),
    ("SAN","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/san/sr_quaternions"),
    ("HAN","experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/han/sr_quaternions"),
    ("OCRP","experiments/Zero_shot_performance_CoNi_x250/ocrp_direct_reynolds_isometric_l4_s42/inference/train_best/sr_quaternions")]),
 "Ti7_zeroshot":("D6h","experiments/Zero_shot_performance_Ti7_deformed/ocrp_direct_reynolds_isometric_l6_s42/inference/train_best/sr_quaternions",
   [("Atindama","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
    ("Q-RBSA","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
    ("QEDSR","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/qedsr/sr_quaternions"),
    ("RCAN","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/rcan/sr_quaternions"),
    ("SAN","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/san/sr_quaternions"),
    ("HAN","experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/han/sr_quaternions"),
    ("OCRP","experiments/Zero_shot_performance_Ti7_deformed/ocrp_direct_reynolds_isometric_l6_s42/inference/train_best/sr_quaternions")]),
 "Ti_indist":("D6h","experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
   [("Atindama","experiments/Ti_Al_1pct/atindama_inpainting_4x4_01/inference/test/sr_quaternions"),
    ("Q-RBSA","experiments/Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01/inference/test/sr_quaternions"),
    ("QEDSR","experiments/Ti_Al_1pct/qedsr_4x4_01/inference/test/sr_quaternions"),
    ("RCAN","experiments/Ti_Al_1pct/rcan_4x4_300ep_01/inference/test/sr_quaternions"),
    ("SAN","experiments/Ti_Al_1pct/san_4x4_300ep_01/inference/test_epoch0205/sr_quaternions"),
    ("HAN","experiments/Ti_Al_1pct/han_4x4_300ep_01/inference/test/sr_quaternions"),
    ("OCRP","experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions")]),
 "Ti64_zeroshot":("D6h","experiments/Zero_shot_performance_Ti64_DIC_Mclean/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
   [("Atindama","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
    ("Q-RBSA","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
    ("QEDSR","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/qedsr/sr_quaternions"),
    ("RCAN","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/rcan/sr_quaternions"),
    ("SAN","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/san/sr_quaternions"),
    ("HAN","experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/han/sr_quaternions"),
    ("OCRP","experiments/Zero_shot_performance_Ti64_DIC_Mclean/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions")]),
}
if __name__=="__main__":
    k=sys.argv[1]; sym,ref,md=REG[k]
    txt=run_dataset(k,sym,ref,md)
    ms=os.environ.get("MINSZ","5")
    print(txt); open(f"analysis/out/finalmetrics_{k}_min{ms}.txt","w").write(txt)
