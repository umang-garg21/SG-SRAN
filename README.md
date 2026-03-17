# Reynolds-QSR: IsoEmbeddingSRAttn-Centered Repository

This repository is focused on one end-to-end workflow:
- Train `IsoEmbeddingSRAttn` for quaternion super-resolution.
- Run inference from LR quaternion maps to SR quaternion maps.
- Visualize LR/SR/HR with IPF maps and layer-wise debug traces.

All legacy code, experiments, and generated artifacts are preserved under `archive/`.

## Core Files

- `models/SR_double_conv_SRattn.py`
- `models/local_iso_embedding.py`
- `training/train_iso_embedding_sr_attn.py`
- `inference/infer_iso_embedding_sr_attn.py`
- `scripts/train_iso_embedding_sr_attn.sh`
- `scripts/infer_iso_embedding_sr_attn.sh`
- `scripts/trace_sr_conv_layers.py`
- `experiments/IN718/iso_embedding_sr_attn_01/config.json`
- `experiments/IN718/iso_embedding_sr_attn_01/config_smoke.json`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
pytest
```

The active test suite is scoped to `tests/` via `pytest.ini` and ignores `archive/`.

## Train

HCP real-data training:

```bash
python -m training.train_iso_embedding_sr_attn \
  --exp_dir experiments/Ti64/iso_embedding_sr_attn_hcp_01 \
  --config config.json \
  --gpu_ids 0
```

Resume:

```bash
python -m training.train_iso_embedding_sr_attn \
  --exp_dir experiments/Ti64/iso_embedding_sr_attn_hcp_01 \
  --config config.json \
  --gpu_ids 0 \
  --resume
```

Smoke run:

```bash
python -m training.train_iso_embedding_sr_attn \
  --exp_dir experiments/Ti64/iso_embedding_sr_attn_hcp_01 \
  --config config_smoke.json \
  --gpu_ids 0
```

IN718 example:

```bash
./scripts/train_iso_embedding_sr_attn.sh experiments/IN718/iso_embedding_sr_attn_01 --config config.json
```

Smoke run:

```bash
./scripts/train_iso_embedding_sr_attn.sh experiments/IN718/iso_embedding_sr_attn_01 --config config_smoke.json
```

## Inference

```bash
./scripts/infer_iso_embedding_sr_attn.sh experiments/IN718/iso_embedding_sr_attn_01 \
  --checkpoint best_model.pt \
  --split Test
```

Inference outputs are written to:
- `experiments/.../inference/<split>/sr_quaternions/*.npy`
- `experiments/.../inference/<split>/ipf/*.png`
- `experiments/.../inference/<split>/summary.json`

## Layer-Wise Debug and Irrep Channel Analysis

Edit `CONFIG` in `scripts/trace_sr_conv_layers.py`, then run:

```bash
python scripts/trace_sr_conv_layers.py
```

This produces:
- Full tensor stats and optional full tensor dumps.
- Spatial channel plots.
- Per-irrep block channel plots after each SR stage.

## Architecture Diagram

Generated diagram:

![IsoEmbeddingSRAttn Architecture](assets/iso_embedding_sr_attn_architecture.png)

Regenerate it with:

```bash
python scripts/make_iso_embedding_sr_attn_architecture.py
```

## Full LaTeX Model and Process Description

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{IsoEmbeddingSRAttn: Model and Training/Inference Process}
\author{}
\date{}

\begin{document}
\maketitle

\section{Problem Setup}
Given low-resolution quaternion orientations
\[
\mathbf{Q}_{\mathrm{LR}} \in \mathbb{R}^{B\times 4\times H\times W},
\]
predict high-resolution orientations
\[
\widehat{\mathbf{Q}}_{\mathrm{SR}} \in \mathbb{R}^{B\times 4\times (rH)\times (rW)}.
\]
Quaternions are scalar-first (\(w,x,y,z\)) and normalized to unit norm.

\section{Local-Iso Embedding Interface}
A crystal-aware local-isometric encoder provides two feature spaces:
\[
E_{a1}: \mathbb{S}^3 \to \mathbb{R}^{d_{a1}}, 
\qquad
E_{\mathrm{full}}: \mathbb{S}^3 \to \mathbb{R}^{d_{\mathrm{full}}}.
\]
These correspond to irreps:
\[
\mathcal{I}_{a1}=\texttt{irreps\_a1},\qquad
\mathcal{I}_{\mathrm{full}}=\texttt{irreps\_full}.
\]
Top-level crystal switch:
\[
\texttt{crystal} \in \{\texttt{fcc},\texttt{hcp}\}
\]
selects \(O\) (FCC) or \(D6\) (HCP) symmetry and matching embedding/symmetry operators.

\section{SR Backbone (No Lift Layer, No HR Conv2)}
Let \(\mathbf{z}_{\mathrm{LR}}=E_{a1}(\mathbf{q}_{\mathrm{LR}})\) (flattened spatially to \(N=HW\) points).  
The SR feature pipeline is:
\begin{align}
\mathbf{f}_1 &= \mathrm{Conv}_{\mathrm{LR1}}^{k=3}(\mathbf{z}_{\mathrm{LR}}),\\
\mathbf{f}_2 &= \mathrm{Conv}_{\mathrm{LR2}}^{k=9}(\mathbf{f}_1),\\
(\mathbf{f}_3,\;H_r,W_r) &= \mathrm{UpConv}^{k=3,r}(\mathbf{f}_2),\\
\mathbf{f}_4 &= \mathrm{Conv}_{\mathrm{HR1}}^{k=3}(\mathbf{f}_3),\\
\mathbf{f}_5 &= \mathrm{AttentionStack}(\mathbf{f}_4),\\
\widehat{\mathbf{z}}_{\mathrm{HR}}^{a1} &= P_{\mathrm{full}\to a1}(\mathbf{f}_5).
\end{align}

\subsection{Layer Irrep Contracts}
\begin{center}
\begin{tabular}{@{}lllll@{}}
\toprule
Stage & Kernel & Irreps In1 & Irreps In2 & Irreps Out \\
\midrule
LR Conv1 & \(3\) & \(\mathcal{I}_{a1}\) & \(\mathcal{I}_{a1}\) & \(\mathcal{I}_{\mathrm{full}}\) \\
LR Conv2 & \(9\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) \\
EquivariantTransposeConv & \(3\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) \\
HR Conv1 & \(3\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) \\
AttentionBlock \(\times M\) & block-local & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) & \(\mathcal{I}_{\mathrm{full}}\) \\
Final Projection & linear & \(\mathcal{I}_{\mathrm{full}}\) & -- & \(\mathcal{I}_{a1}\) \\
\bottomrule
\end{tabular}
\end{center}

\subsection{EquivariantSpatialConv}
For each spatial position \(i\), with neighborhood \(\mathcal{N}_k(i)\):
\begin{align}
\mathbf{c}_i &= \sum_{j\in\mathcal{N}_k(i)} w_{ij}\mathbf{x}_j,\qquad \sum_j w_{ij}=1,\\
\mathbf{y}_i &= \mathrm{TP}(\mathbf{x}_i,\mathbf{c}_i)+\mathbf{r}_i.
\end{align}
\(\mathrm{TP}\) is \texttt{FullyConnectedTensorProduct} with specified irreps.  
For LR Conv1, residual is disabled; for full\(\to\)full layers residual is additive identity.

\subsection{EquivariantTransposeConv}
Depthwise transposed convolution upsamples by factor \(r\), initialized with bilinear kernel, then applies tensor-product mixing with local context:
\[
\mathbf{u}=\mathrm{DepthwiseConvTranspose}(\mathbf{x}),\quad
\mathbf{y}_i=\mathrm{TP}(\mathbf{u}_i,\mathbf{c}_i)+\mathbf{r}_i.
\]

\subsection{Block-Local Equivariant Attention}
Within each \(b_h\times b_w\) block (\(N_b=b_hb_w\)):
\begin{align}
s_{ij} &= \exp(\lambda)\,\langle \bar{\mathbf{x}}_i,\bar{\mathbf{x}}_j\rangle
+ \beta(\mathbf{p}_i)+\beta(\mathbf{p}_j),\\
\alpha_{ij} &= \mathrm{softmax}_j(s_{ij}),\\
\mathbf{h}_i &= L_{\mathrm{in}}(\mathbf{x}_i),\\
\mathbf{v}_i &= \mathrm{TP}_{\mathrm{val}}(\mathbf{h}_i,\mathbf{sh}_i), \quad
\mathbf{sh}_i\in \mathrm{Irreps}(1\times 0e + 1\times 2e),\\
\mathbf{c}_i &= \sum_j \alpha_{ij}\mathbf{v}_j,\\
\Delta \mathbf{x}_i &= L_{\mathrm{out}}\!\Big(\mathrm{TP}_{\mathrm{out}}(\mathbf{h}_i,\mathbf{c}_i)\Big),\\
\mathbf{x}_i &\leftarrow \mathbf{x}_i+\Delta \mathbf{x}_i.
\end{align}
\(L_{\mathrm{out}}\) is zero-initialized at start.

\section{Decoder: Cubochoric-Sampled Feature Optimization}
Decoder target irreps are \(a1\) (final output feature space).

\subsection{Seed Table}
Sample FZ quaternions \(\{\mathbf{q}_t\}_{t=1}^T\) (cubochoric), precompute:
\[
\mathbf{z}_t = E_{a1}(\mathbf{q}_t).
\]

\subsection{Nearest-Seed + Refinement}
For target feature \(\mathbf{z}\), choose top-\(K\) nearest seeds by squared distance
\[
d_t=\|\mathbf{z}-\mathbf{z}_t\|_2^2.
\]
Initialize \(\mathbf{u}_k^{(0)}=\mathbf{q}_{t_k}\), optimize:
\[
\min_{\{\mathbf{u}_k\}_{k=1}^K}
\frac{1}{K}\sum_{k=1}^K
\left\|
E_{a1}\!\left(\mathrm{norm}(\mathbf{u}_k)\right)-\mathbf{z}
\right\|_2^2
\]
with Adam for fixed steps, then select best \(k^\star\).

\subsection{Fundamental-Zone Reduction}
Given symmetry operators \(\{s_g\}\),
\[
\mathbf{q}^{(g)}=\mathrm{norm}(s_g^{-1}\otimes \mathbf{q}),\qquad
g^\star=\arg\max_g |w^{(g)}|,
\]
output \(\mathbf{q}_{\mathrm{FZ}}=\mathbf{q}^{(g^\star)}\).

\section{Training Objective}
Training is feature-supervision in \(a1\) space:
\begin{align}
\mathbf{z}_{\mathrm{LR}} &= E_{a1}(\mathbf{Q}_{\mathrm{LR}}),\\
\mathbf{z}_{\mathrm{HR}} &= E_{a1}(\mathbf{Q}_{\mathrm{HR}}),\\
\widehat{\mathbf{z}}_{\mathrm{HR}} &= F_{\mathrm{SR}}(\mathbf{z}_{\mathrm{LR}}),\\
\mathcal{L}_{\mathrm{SR}} &= \frac{1}{BN_{\mathrm{HR}}}
\sum_{b,n}
\left\|
\widehat{\mathbf{z}}_{\mathrm{HR},b,n}
-
\mathbf{z}_{\mathrm{HR},b,n}
\right\|_2^2.
\end{align}
\(E_{a1}\) targets are computed without gradient flow (detached).

\section{Inference}
\[
\widehat{\mathbf{Q}}_{\mathrm{SR}}=
\mathrm{Decode}_{a1}\!\left(
P_{\mathrm{full}\to a1}\!\left(
\mathrm{Attention}\!\left(
\mathrm{Conv}_{\mathrm{HR1}}\!\left(
\mathrm{UpConv}\!\left(
\mathrm{Conv}_{\mathrm{LR2}}\!\left(
\mathrm{Conv}_{\mathrm{LR1}}\!\left(
E_{a1}(\mathbf{Q}_{\mathrm{LR}})
\right)\right)\right)\right)\right)\right)\right).
\]

\begin{algorithm}[H]
\caption{One Training Iteration}
\begin{algorithmic}[1]
\State Input batch: \(\mathbf{Q}_{\mathrm{LR}},\mathbf{Q}_{\mathrm{HR}}\)
\State Normalize quaternions
\State \(\mathbf{z}_{\mathrm{LR}}\gets E_{a1}(\mathbf{Q}_{\mathrm{LR}})\) (detach)
\State \(\mathbf{z}_{\mathrm{HR}}\gets E_{a1}(\mathbf{Q}_{\mathrm{HR}})\) (detach)
\State \(\widehat{\mathbf{z}}_{\mathrm{HR}}\gets F_{\mathrm{SR}}(\mathbf{z}_{\mathrm{LR}})\)
\State \(\mathcal{L}\gets \mathrm{MSE}(\widehat{\mathbf{z}}_{\mathrm{HR}},\mathbf{z}_{\mathrm{HR}})\)
\State Backpropagate \(\mathcal{L}\), gradient-clip, optimizer step
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[H]
\caption{Inference (LR \(\to\) SR Quaternion Map)}
\begin{algorithmic}[1]
\State Input \(\mathbf{Q}_{\mathrm{LR}}\), shape \((H,W)\)
\State \(\mathbf{z}_{\mathrm{LR}}\gets E_{a1}(\mathbf{Q}_{\mathrm{LR}})\)
\State \(\widehat{\mathbf{z}}_{\mathrm{HR}}\gets F_{\mathrm{SR}}(\mathbf{z}_{\mathrm{LR}})\)
\State \(\widehat{\mathbf{Q}}_{\mathrm{raw}}\gets \mathrm{Decoder}_{a1}(\widehat{\mathbf{z}}_{\mathrm{HR}})\)
\State \(\widehat{\mathbf{Q}}_{\mathrm{SR}}\gets \mathrm{ReduceToFZ}(\widehat{\mathbf{Q}}_{\mathrm{raw}})\)
\State Return \(\widehat{\mathbf{Q}}_{\mathrm{SR}}\)
\end{algorithmic}
\end{algorithm}

\end{document}
```
