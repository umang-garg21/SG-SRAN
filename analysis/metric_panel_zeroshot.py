#!/usr/bin/env python3
"""Physics-grounded metric panel for orientation-map SR, vs per-pixel mean misorientation.
Metrics (symmetry-aware, HCP D6): strict mean; +/-1px tolerant mean; interior/boundary split;
tolerant boundary-F1; grain-count ratio + grain-size Wasserstein; basal (0001) pole-figure L1."""
import os, sys, glob
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.symmetry_utils import proper_symmetry_quaternions

SYM = proper_symmetry_quaternions("D6h").astype(np.float32)  # (12,4) scalar-first

def qmul(a, b):
    aw,ax,ay,az=a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)
def norm(q): return q/np.clip(np.linalg.norm(q,axis=-1,keepdims=True),1e-12,None)

def mis_deg(pred, tgt):
    """per-pixel symmetry-aware misorientation (deg), pred,tgt: (H,W,4)."""
    H,W=pred.shape[:2]; p=norm(pred.reshape(-1,4)); t=norm(tgt.reshape(-1,4))
    eq=qmul(SYM[:,None,:], t[None,:,:])
    best=np.clip(np.abs((eq*p[None,:,:]).sum(-1)).max(0),0,1)
    return np.rad2deg(2*np.arccos(best)).reshape(H,W)

def tol_mis_deg(pred, tgt, r=1):
    """per-pixel min misorientation to HR within (2r+1)^2 window (registration-tolerant)."""
    H,W=pred.shape[:2]
    best=np.full((H,W), 999.0, np.float32)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            t=np.roll(np.roll(tgt,dy,0),dx,1)
            best=np.minimum(best, mis_deg(pred,t))
    return best

def boundary_mask(q, thr=5.0):
    b=np.zeros(q.shape[:2],bool)
    h=mis_deg(q[:,:-1],q[:,1:])>thr; b[:,:-1]|=h; b[:,1:]|=h
    v=mis_deg(q[:-1],q[1:])>thr; b[:-1]|=v; b[1:]|=v
    return b

def dilate(m):
    d=m.copy()
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            d|=np.roll(np.roll(m,dy,0),dx,1)
    return d

def tolerant_bf1(pred_b, ref_b):
    ref_d=dilate(ref_b); pred_d=dilate(pred_b)
    tp_p=(pred_b&ref_d).sum(); prec=tp_p/max(pred_b.sum(),1)
    tp_r=(ref_b&pred_d).sum(); rec=tp_r/max(ref_b.sum(),1)
    return 2*prec*rec/max(prec+rec,1e-9)

def grains(q, thr=5.0):
    H,W=q.shape[:2]; N=H*W; idx=np.arange(N).reshape(H,W)
    ii=[]; jj=[]
    same_h=mis_deg(q[:,:-1],q[:,1:])<=thr
    a=idx[:,:-1][same_h]; b=idx[:,1:][same_h]; ii+=[a]; jj+=[b]
    same_v=mis_deg(q[:-1],q[1:])<=thr
    a=idx[:-1][same_v]; b=idx[1:][same_v]; ii+=[a]; jj+=[b]
    ii=np.concatenate(ii); jj=np.concatenate(jj)
    g=coo_matrix((np.ones_like(ii),(ii,jj)),shape=(N,N))
    n,lab=connected_components(g,directed=False)
    _,counts=np.unique(lab,return_counts=True)
    return n, counts

def basal_pf_hist(q, nb=24):
    """(0001) c-axis pole figure, Lambert equal-area 2D histogram, upper hemisphere."""
    p=norm(q.reshape(-1,4)); w,x,y,z=p[:,0],p[:,1],p[:,2],p[:,3]
    # sample-frame image of crystal c=[0,0,1] = 3rd column of R(q)
    vx=2*(x*z+w*y); vy=2*(y*z-w*x); vz=1-2*(x*x+y*y)
    s=np.sign(vz); s[s==0]=1; vx*=s; vy*=s; vz=np.abs(vz)      # fold to upper hemisphere
    # Lambert equal-area projection of (vx,vy,vz)
    f=np.sqrt(np.clip(2/(1+vz),0,None)); X=f*vx; Y=f*vy
    h,_,_=np.histogram2d(X,Y,bins=nb,range=[[-1.5,1.5],[-1.5,1.5]])
    return h/max(h.sum(),1)

def eval_method(sr_dir, ref_dir, n):
    strict=[]; tol=[]; inter=[]; bound=[]; bf1=[]; gratio=[]; gwass=[]; pf=[]
    for i in range(n):
        tag=f"sample_{i:06d}"
        sr=np.load(f"{sr_dir}/{tag}_sr.npy").astype(np.float32)
        hr=np.load(f"{ref_dir}/{tag}_hr.npy").astype(np.float32)
        m=mis_deg(sr,hr); strict.append(m.mean())
        tol.append(tol_mis_deg(sr,hr,1).mean())
        hb=boundary_mask(hr); band=dilate(hb)
        inter.append(m[~band].mean()); bound.append(m[band].mean() if band.any() else np.nan)
        bf1.append(tolerant_bf1(boundary_mask(sr),hb))
        ng_sr,c_sr=grains(sr); ng_hr,c_hr=grains(hr)
        gratio.append(ng_sr/max(ng_hr,1))
        gwass.append(wasserstein_distance(np.log10(c_sr),np.log10(c_hr)))
        pf.append(np.abs(basal_pf_hist(sr)-basal_pf_hist(hr)).sum())  # L1 (TV*2), [0,2]
    f=lambda a: float(np.nanmean(a))
    return dict(strict_mean=f(strict), tol1_mean=f(tol), interior=f(inter), boundary=f(bound),
                bf1_tol=f(bf1), grain_ratio=f(gratio), grain_wass=f(gwass), basal_pf_L1=f(pf))

if __name__ == "__main__":
    R="experiments/Zero_shot_performance_Ti64_DIC_Mclean"
    LB=f"{R}/learned_baselines_4x4"
    ref=f"{R}/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions"
    methods=[("Atindama",f"{LB}/atindama_inpainting/sr_quaternions"),
             ("Q-RBSA",f"{LB}/qrbsaadapted/sr_quaternions"),
             ("QEDSR",f"{LB}/qedsr/sr_quaternions"),
             ("RCAN",f"{LB}/rcan/sr_quaternions"),
             ("SAN",f"{LB}/san/sr_quaternions"),
             ("HAN",f"{LB}/han/sr_quaternions"),
             ("OCRP",ref)]
    n=len(glob.glob(f"{ref}/*_sr.npy"))
    print(f"Ti64 zero-shot, n={n} patches (HCP D6)\n")
    cols=["strict_mean","tol1_mean","interior","boundary","bf1_tol","grain_ratio","grain_wass","basal_pf_L1"]
    hdr=["lower_better"]+["v"]*0
    print(f"{'method':10s} "+" ".join(f"{c:>12s}" for c in cols))
    better={"bf1_tol":"high"}  # rest lower-better except grain_ratio (closer to 1)
    rows={}
    for name,d in methods:
        rows[name]=eval_method(d,ref,n)
        r=rows[name]
        print(f"{name:10s} "+" ".join(f"{r[c]:12.3f}" for c in cols))
    print("\n(lower better: strict_mean,tol1_mean,interior,boundary,grain_wass,basal_pf_L1;"
          " higher better: bf1_tol; grain_ratio: closer to 1.0 = better)")
