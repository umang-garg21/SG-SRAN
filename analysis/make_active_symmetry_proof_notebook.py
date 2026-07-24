from __future__ import annotations

import json
from pathlib import Path


OUT = Path("analysis/IN718_OCRP_4x4_active_right_invariant_left_SO3_equivariant_layer_proof.ipynb")

_CELL_COUNTER = 0


def _next_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"cell-{_CELL_COUNTER:03d}"


def md(source: str) -> dict:
    return {
        "id": _next_id(),
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "id": _next_id(),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# IN718 4×4 OCRP anchorless — active-convention right-invariance and left-SO(3)-equivariance proof

This notebook checks the trained **IN718 4×4 OCRP anchorless epoch-24** module layer by layer.

The claim being tested is deliberately side-specific:

$$
\boxed{\text{right crystal symmetry: } q \mapsto q \otimes h,\ h\in O \quad\Rightarrow\quad \text{invariance}}
$$

$$
\boxed{\text{left global rotation: } q \mapsto g \otimes q,\ g\in SO(3) \quad\Rightarrow\quad \text{SO(3)-equivariance}}
$$

and **not vice versa**.

In active convention, the orientation quaternion maps a crystal-frame vector into the sample/lab frame. Therefore:

- multiplying on the **right** changes only the crystal-basis representative: same physical orientation class;
- multiplying on the **left** rotates the entire physical orientation in the sample/lab frame.

So the correct quotient/equivariance behavior is:

$$
F(q \otimes h)=F(q), \qquad h\in O,
$$

and for irrep-valued feature tensors,

$$
F(g\otimes q)=D(g)F(q).
$$

Because tensors are stored as row vectors in this code, the notebook applies the row-vector version:

$$
F(g\otimes q)_{\mathrm{row}} = F(q)_{\mathrm{row}}D(g)^\mathsf{T}.
$$

The checks below use real IN718 validation data and the real epoch-24 checkpoint.
        """
    ),
    md(
        r"""
## Exact source locations used in this proof

This notebook is not proving a toy model. It calls the same production source modules used by the trained OCRP checkpoint.

| What is being checked | Exact source lines |
| --- | --- |
| Experiment says this is 4×4 FCC/O(h), full irreps, 9×9 window, 6 slots, 2° clustering | [config_new.json:16](/data/home/umang/Materials/Reynolds-QSR_paper/experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/config_new.json:16), lines 16-37 |
| Experiment selects `models.SR_4x4_from_4x1_ocrp_anchorless.IsoEmbedding4x4FromSROCRP` and representative `x_230` | [config_new.json:95](/data/home/umang/Materials/Reynolds-QSR_paper/experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/config_new.json:95), lines 95-100 |
| 4×4 wrapper fixes `upsample_factor=(4, 4)` | [SR_4x4_from_4x1_ocrp_anchorless.py:2215](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2215), lines 2215-2219 |
| Model builds local-iso encoder | [SR_4x4_from_4x1_ocrp_anchorless.py:1785](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1785), lines 1785-1790 |
| Model builds LR conv, OCRP, HR conv1, HR conv2 | [SR_4x4_from_4x1_ocrp_anchorless.py:1870](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1870), lines 1870-1936 |
| Public encoder dispatches to full local-iso irreps | [SR_4x4_from_4x1_ocrp_anchorless.py:2016](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2016), lines 2016-2019 |
| Wrapper passes API quaternions into `forward_irreps_passive` | [SR_4x4_from_4x1_ocrp_anchorless.py:490](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:490), lines 490-492 |
| Local-iso source converts passive input to active internally | [local_iso_embedding.py:635](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:635), lines 635-649 |
| Local-iso source builds active rotation matrices | [local_iso_embedding.py:24](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:24), lines 24-59 |
| Local-iso source performs orbit averaging and irrep projection | [local_iso_embedding.py:586](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:586), lines 586-617 |
| Feature SR path applies LR conv → OCRP → HR convs and returns probes | [SR_4x4_from_4x1_ocrp_anchorless.py:2031](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2031), lines 2031-2086 |
| Cosine-masked equivariant convolution | [SR_4x4_from_4x1_ocrp_anchorless.py:315](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:315), lines 315-441 |
| OCRP forward: bank → cluster → slots → router → cross-attention → HR assembly | [SR_4x4_from_4x1_ocrp_anchorless.py:1584](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1584), lines 1584-1698 |
| Quaternion clusterer uses symmetry-aware misorientation edges | [SR_4x4_from_4x1_ocrp_anchorless.py:800](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:800), lines 800-819 |
| Cluster slot builder emits scalar slot masks/metadata | [SR_4x4_from_4x1_ocrp_anchorless.py:852](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:852), lines 852-1008 |
| Patch router maps scalar slot masks + phase to scalar logits | [SR_4x4_from_4x1_ocrp_anchorless.py:1037](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1037), lines 1037-1166 |
| Anchorless member cross-attention uses invariant summaries for weights and weighted sums of equivariant features | [SR_4x4_from_4x1_ocrp_anchorless.py:1225](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1225), lines 1225-1333 |
        """
    ),
    md(
        r"""
## Why the side matters

Let \(R(q)\) be the active rotation matrix represented by the active quaternion \(q\).

For active quaternions:

$$
R(g\otimes q)=R(g)R(q),
\qquad
R(q\otimes h)=R(q)R(h).
$$

The local-iso encoder averages tensor powers over a crystal orbit:

$$
T(R)=\frac{1}{|O|}\sum_{u\in O\cdot u_0}(Ru)^{\otimes k}.
$$

Right multiplication by \(h\in O\) gives:

$$
T(RR(h))
=
\frac{1}{|O|}\sum_{u\in O\cdot u_0}(RR(h)u)^{\otimes k}.
$$

Because \(h\) is a crystal symmetry, \(R(h)\) only permutes the orbit \(O\cdot u_0\). The average is unchanged:

$$
T(RR(h))=T(R).
$$

Left multiplication by \(g\in SO(3)\) gives:

$$
T(R(g)R)=D(g)T(R),
$$

after projection to the irreducible representation channels.

That is the root of the whole notebook:

- **right by crystal group** = same quotient representative = invariant;
- **left by any SO(3) rotation** = physical rotation of the lab-frame tensor = equivariant.
        """
    ),
    code(
        r"""
from __future__ import annotations

import inspect
import importlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from e3nn.o3 import Irreps
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This notebook must be run in the project environment with e3nn installed. "
        "On this machine, use /data/home/umang/miniconda3/envs/material/bin/python "
        "or the matching Jupyter kernel."
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/data/home/umang/Materials/Reynolds-QSR_paper")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/config_new.json"
CHECKPOINT_PATH = ROOT / "experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/checkpoints/epoch_0024.pt"
LR_SAMPLE_PATH = Path("/data/home/umang/Materials/Materials_data_mount/datasets/IN718_QSR_x4/Val/LR_Data/IN718_QSR_x4_val_lr_x_block_230.npy")

OUT_DIR = ROOT / "analysis/out/ocrp_4x4_active_symmetry_layer_proof"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
DTYPE = torch.float32
LR_R0, LR_C0 = 20, 26
LR_CROP_SIZE = 4

print("config:", CONFIG_PATH)
print("checkpoint:", CHECKPOINT_PATH)
print("sample:", LR_SAMPLE_PATH)
print("out:", OUT_DIR)
        """
    ),
    code(
        r"""
def qnormalize_torch(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def qconj_torch(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def qmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Hamilton product a ⊗ b for scalar-first [w,x,y,z] quaternions.
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    out = torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
    return qnormalize_torch(out)


def quat_to_matrix_active_torch(q: torch.Tensor) -> torch.Tensor:
    # Same active scalar-first quaternion-to-matrix formula used by local_iso_embedding.py:24-59.
    q = qnormalize_torch(q)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def qnormalize_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)
    return np.where(q[..., :1] < 0.0, -q, q).astype(np.float32)


def qconj_np(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float32, copy=True)
    out[..., 1:] *= -1.0
    return out


def storage_to_active_np(q_storage: np.ndarray) -> np.ndarray:
    return qnormalize_np(qconj_np(q_storage))


def checkpoint_input_from_active(q_active: torch.Tensor) -> torch.Tensor:
    # Notebook adapter: active display convention -> unchanged checkpoint API convention.
    return qnormalize_torch(qconj_torch(q_active))


def checkpoint_output_to_active(q_api: torch.Tensor) -> torch.Tensor:
    return qnormalize_torch(qconj_torch(q_api))


def right_crystal_action_active(q_active: torch.Tensor, h_active: torch.Tensor) -> torch.Tensor:
    # Active right crystal action: q -> q ⊗ h.
    return qmul_torch(q_active, h_active)


def left_so3_action_active(g_active: torch.Tensor, q_active: torch.Tensor) -> torch.Tensor:
    # Active left SO(3) action: q -> g ⊗ q.
    return qmul_torch(g_active, q_active)


def apply_feature_action_row(t: torch.Tensor, A_row: torch.Tensor) -> torch.Tensor:
    # Apply row-vector irrep action to the last feature dimension.
    return torch.einsum("...c,cd->...d", t, A_row)
        """
    ),
    md(
        r"""
## Load the real epoch-24 model and real IN718 crop

The checkpoint API still uses the historical storage/passive input form. This notebook displays and manipulates active quaternions only, then uses:

$$
q_{\mathrm{api}} = q_{\mathrm{active}}^*
$$

at the model boundary.

That is consistent with the source because `forward_irreps_passive` immediately computes:

$$
q_{\mathrm{active}} = q_{\mathrm{passive}}^*
$$

at [local_iso_embedding.py:647](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:647), and then builds the active rotation matrix at [local_iso_embedding.py:648](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:648).
        """
    ),
    code(
        r"""
def build_model_from_config(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = json.loads(config_path.read_text())
    model_cfg = cfg["model"]
    module = importlib.import_module(model_cfg["model_module"])
    cls = getattr(module, model_cfg["model_class"])

    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in cfg.items() if k in sig.parameters}
    kwargs["device"] = device
    kwargs["decoder_eager_init"] = False

    model = cls(**kwargs).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    load_result = model.load_state_dict(state, strict=True)
    return cfg, ckpt, model, load_result


cfg, ckpt, model, load_result = build_model_from_config(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)
lr_storage = np.load(LR_SAMPLE_PATH)
lr_active = storage_to_active_np(lr_storage)

lr_crop_active_np = lr_active[
    LR_R0 : LR_R0 + LR_CROP_SIZE,
    LR_C0 : LR_C0 + LR_CROP_SIZE,
]
lr_crop_active = torch.from_numpy(lr_crop_active_np).to(device=DEVICE, dtype=DTYPE)
lr_shape = (LR_CROP_SIZE, LR_CROP_SIZE)
num_lr_tokens = LR_CROP_SIZE * LR_CROP_SIZE

print("checkpoint saved epoch field:", ckpt.get("epoch"), "(file is epoch_0024.pt)")
print("load result:", load_result)
print("model class:", model.__class__.__name__)
print("feature_irreps:", model.feature_irreps)
print("irreps_feat:", model.irreps_feat)
print("feature_dim:", model.feature_dim)
print("upsample_factor:", model.upsample_factor)
print("OCRP window_size:", model.ocrp.window_size)
print("OCRP kmax_slots:", model.ocrp.kmax_slots)
print("LR crop active shape:", tuple(lr_crop_active.shape), "global top-left:", (LR_R0, LR_C0))
        """
    ),
    md(
        r"""
## Build the two active-convention probes

We build three branches:

1. `base`: original active crop \(q\);
2. `right`: independently apply random cubic crystal symmetries at each LR pixel:

   $$
   q_i \mapsto q_i\otimes h_i,\qquad h_i\in O;
   $$

   this is a strong right-invariance test because every pixel can choose a different crystal representative.

3. `left`: apply one generic non-cubic SO(3) rotation to the whole crop:

   $$
   q_i \mapsto g\otimes q_i,\qquad g\in SO(3).
   $$

For left equivariance, the row-vector feature action is:

$$
A_{\mathrm{row}}(g)=D_{\mathrm{e3nn}}(R(g))^\mathsf{T}.
$$

The notebook verifies that \(A_{\mathrm{row}}\) is orthogonal and then uses it for every irrep-valued layer.
        """
    ),
    code(
        r"""
sym_ops_active = model.encoder.sym_ops.detach().to(device=DEVICE, dtype=DTYPE)

# Independent right crystal representative per LR pixel.
gen = torch.Generator(device="cpu")
gen.manual_seed(123)
sym_idx = torch.randint(0, sym_ops_active.shape[0], (num_lr_tokens,), generator=gen)
h_per_pixel = sym_ops_active[sym_idx].reshape(LR_CROP_SIZE, LR_CROP_SIZE, 4)
lr_crop_right_active = right_crystal_action_active(lr_crop_active, h_per_pixel)

# One fixed generic SO(3) probe, intentionally not one of the cubic symmetries.
g_left = qnormalize_torch(torch.tensor([0.913, 0.143, 0.287, 0.258], dtype=DTYPE, device=DEVICE).view(1, 4))[0]
lr_crop_left_active = left_so3_action_active(g_left.view(1, 1, 4).expand_as(lr_crop_active), lr_crop_active)

irreps_feat = Irreps(model.irreps_feat)
D_col = irreps_feat.D_from_matrix(quat_to_matrix_active_torch(g_left).detach().cpu()).to(device=DEVICE, dtype=DTYPE)
A_left_row = D_col.T.contiguous()
orth_err = float((A_left_row.T @ A_left_row - torch.eye(A_left_row.shape[0], device=DEVICE, dtype=DTYPE)).abs().max())

print("symmetry ops:", tuple(sym_ops_active.shape))
print("right branch uses per-token sym op indices:", sym_idx.reshape(LR_CROP_SIZE, LR_CROP_SIZE).tolist())
print("left SO(3) probe:", g_left.tolist())
print("row action matrix shape:", tuple(A_left_row.shape))
print("row action orthogonality max error:", f"{orth_err:.3e}")
        """
    ),
    md(
        r"""
## Run the real OCRP feature pipeline

This function is intentionally thin. It only:

1. converts active notebook quaternions to the unchanged checkpoint API convention;
2. calls `model.encode(...)`;
3. calls `model._forward_sr_features(..., return_aux=True)`.

The layer tensors are produced by source code at:

- `model.encode`: [SR_4x4_from_4x1_ocrp_anchorless.py:2016](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2016), lines 2016-2019;
- `_forward_sr_features`: [SR_4x4_from_4x1_ocrp_anchorless.py:2031](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2031), lines 2031-2086;
- OCRP aux tensors: [SR_4x4_from_4x1_ocrp_anchorless.py:1661](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1661), lines 1661-1692.
        """
    ),
    code(
        r"""
def run_feature_pipeline_from_active(q_active_grid: torch.Tensor):
    q_active_flat = q_active_grid.reshape(-1, 4)
    q_api_flat = checkpoint_input_from_active(q_active_flat)
    with torch.no_grad():
        feat_lr_flat = model.encode(q_api_flat)
        feat_lr = feat_lr_flat.reshape(1, -1, feat_lr_flat.shape[-1])
        feat_hr, hr_shape_out, aux = model._forward_sr_features(
            lr_quats=q_api_flat.reshape(1, -1, 4),
            feat_lr=feat_lr,
            lr_shape=lr_shape,
            return_aux=True,
        )
    tensors = {
        "feat_lr_encode": feat_lr,
        **aux,
        "feat_hr_final": feat_hr,
    }
    return tensors, hr_shape_out


base, hr_shape = run_feature_pipeline_from_active(lr_crop_active)
right, hr_shape_right = run_feature_pipeline_from_active(lr_crop_right_active)
left, hr_shape_left = run_feature_pipeline_from_active(lr_crop_left_active)

assert hr_shape == hr_shape_right == hr_shape_left
print("HR shape:", hr_shape)
print("available tensors:")
for k in sorted(base):
    v = base[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k:26s} {tuple(v.shape)} {v.dtype}")
    elif k == "probe_stages":
        print(f"  {k:26s} list length={len(v)}")
        for stage in v:
            feat = stage.get("feat")
            shape = tuple(feat.shape) if isinstance(feat, torch.Tensor) else None
            print(f"    - {stage.get('name')}: feat={shape}, image_shape={stage.get('shape')}")
    else:
        print(f"  {k:26s} {type(v).__name__}: {v}")
        """
    ),
    md(
        r"""
## Metrics used in the layerwise proof

For an irrep-valued feature tensor \(X\), the checks are:

Right crystal invariance:

$$
\mathrm{err}_{\mathrm{right}}=\|X(q\otimes h)-X(q)\|.
$$

Left SO(3) equivariance:

$$
\mathrm{err}_{\mathrm{left}}=\|X(g\otimes q)-X(q)A_{\mathrm{row}}(g)\|.
$$

Wrong-side control:

$$
\mathrm{err}_{\mathrm{left\ as\ invariant}}=\|X(g\otimes q)-X(q)\|.
$$

That last number should be large for non-scalar irrep features. It proves we are not accidentally treating left SO(3) as an invariance.

For scalar routing quantities such as `cluster_ids`, `slot_mask`, `slot_meta`, `router_logits`, and `owner_idx`, there is no irrep channel action. They are type-0 quantities, so they should be invariant under both the right crystal representative change and the common left frame rotation. This does not violate the main claim: those scalar quantities only choose/weight equivariant feature vectors.

One nuance: `xattn_alpha` is a returned diagnostic softmax tensor from the attention path. It is scalar, but it is not expected to be bit-exact because tiny floating-point differences in nearly tied scores can move softmax weights slightly. The important functional check is that the scalar-weighted feature outputs `patch_prop`, `patch_out`, and the final HR feature tensor still satisfy the symmetry relations.
        """
    ),
    code(
        r"""
def tensor_error(expected: torch.Tensor, observed: torch.Tensor) -> dict:
    if expected.dtype == torch.bool or observed.dtype == torch.bool:
        eq = expected == observed
        return {
            "kind": "exact/bool",
            "rel": np.nan,
            "rms": float((expected.float() - observed.float()).square().mean().sqrt()),
            "max_abs": float((expected.float() - observed.float()).abs().max()),
            "exact": bool(eq.all()),
            "neq_frac": float((~eq).float().mean()),
        }
    if not torch.is_floating_point(expected) or not torch.is_floating_point(observed):
        eq = expected == observed
        return {
            "kind": "exact/int",
            "rel": np.nan,
            "rms": float((expected.float() - observed.float()).square().mean().sqrt()),
            "max_abs": float((expected.float() - observed.float()).abs().max()),
            "exact": bool(eq.all()),
            "neq_frac": float((~eq).float().mean()),
        }
    diff = (expected - observed).detach().float()
    rel = float(diff.norm().item() / (observed.detach().float().norm().item() + 1e-12))
    return {
        "kind": "float",
        "rel": rel,
        "rms": float(diff.square().mean().sqrt()),
        "max_abs": float(diff.abs().max()),
        "exact": bool(diff.abs().max() == 0),
        "neq_frac": np.nan,
    }


feature_tensors = [
    "feat_lr_encode",
    "feat_lr_pre_ocrp",
    "bank_f",
    "patch_prop",
    "patch_out",
    "feat_hr_raw_ocrp",
    "feat_hr_post_hr_conv1",
    "feat_hr_post_hr_conv2",
    "feat_hr_post_hr_conv",
    "feat_hr_final",
]

scalar_tensors = [
    "cluster_ids",
    "slot_mask",
    "slot_valid",
    "slot_meta",
    "router_logits",
    "owner_idx",
    "xattn_alpha",
]

rows = []
for name in feature_tensors:
    if name not in base or base[name] is None:
        continue
    right_err = tensor_error(base[name], right[name])
    left_eq_err = tensor_error(apply_feature_action_row(base[name], A_left_row), left[name])
    left_inv_wrong = tensor_error(base[name], left[name])
    rows.append(
        {
            "tensor": name,
            "tensor_type": "irrep feature",
            "shape": tuple(base[name].shape),
            "right_crystal_should_be": "invariant",
            "right_rel": right_err["rel"],
            "right_rms": right_err["rms"],
            "right_max_abs": right_err["max_abs"],
            "left_SO3_should_be": "equivariant",
            "left_eq_rel": left_eq_err["rel"],
            "left_eq_rms": left_eq_err["rms"],
            "left_eq_max_abs": left_eq_err["max_abs"],
            "wrong_left_as_invariant_rel": left_inv_wrong["rel"],
            "wrong_left_as_invariant_rms": left_inv_wrong["rms"],
            "exact": np.nan,
            "neq_frac": np.nan,
        }
    )

for name in scalar_tensors:
    if name not in base or base[name] is None:
        continue
    right_err = tensor_error(base[name], right[name])
    left_inv_err = tensor_error(base[name], left[name])
    rows.append(
        {
            "tensor": name,
            "tensor_type": "scalar routing/index",
            "shape": tuple(base[name].shape),
            "right_crystal_should_be": "invariant",
            "right_rel": right_err["rel"],
            "right_rms": right_err["rms"],
            "right_max_abs": right_err["max_abs"],
            "left_SO3_should_be": "scalar invariant",
            "left_eq_rel": left_inv_err["rel"],
            "left_eq_rms": left_inv_err["rms"],
            "left_eq_max_abs": left_inv_err["max_abs"],
            "wrong_left_as_invariant_rel": np.nan,
            "wrong_left_as_invariant_rms": np.nan,
            "exact": left_inv_err["exact"] and right_err["exact"],
            "neq_frac": max(
                0.0 if np.isnan(right_err["neq_frac"]) else right_err["neq_frac"],
                0.0 if np.isnan(left_inv_err["neq_frac"]) else left_inv_err["neq_frac"],
            ),
        }
    )

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "layerwise_right_invariant_left_equivariant_checks.csv"
df.to_csv(csv_path, index=False)
print("saved:", csv_path)
display(
    df[
        [
            "tensor",
            "tensor_type",
            "shape",
            "right_rel",
            "right_max_abs",
            "left_eq_rel",
            "left_eq_max_abs",
            "wrong_left_as_invariant_rel",
            "exact",
        ]
    ]
)
        """
    ),
    md(
        r"""
## Does the side still matter after the encoder?

Yes — but in two different ways.

The encoder output is **not simply invariant**. It is:

$$
E(q\otimes h)=E(q),\qquad h\in O,
$$

but also:

$$
E(g\otimes q)=E(q)A_{\mathrm{row}}(g),\qquad g\in SO(3).
$$

So after the encoder:

- right crystal invariance is already built into the irrep feature itself;
- left SO(3) is still a nontrivial channel transformation.

This means:

### For right crystal invariance

If a downstream block used **only** the encoded feature \(E(q)\), then right crystal invariance would be automatic for any deterministic function:

$$
E(q\otimes h)=E(q)
\quad\Rightarrow\quad
N(E(q\otimes h))=N(E(q)).
$$

But OCRP also uses a raw quaternion support bank `bank_q` for clustering. Therefore the network must still ensure that this raw-quaternion branch uses only right-crystal-invariant decisions. In the source, that is done by symmetry-aware misorientation in the clusterer at [SR_4x4_from_4x1_ocrp_anchorless.py:817](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:817), lines 817-819.

### For left SO(3) equivariance

Equivariance is **not automatic** after encoding. A generic learned latent map \(N\) must satisfy:

$$
N(XA_{\mathrm{row}}(g)) = N(X)A_{\mathrm{row}}(g).
$$

An arbitrary dense MLP or arbitrary dense linear projection on the 14 latent channels will generally not satisfy this. It would mix the \(l=2\) and \(l=4\) irrep channels in a way that does not commute with \(D(g)\), so left equivariance would be broken.

That is why the network still has to be designed carefully after the encoder:

- use equivariant operations on irrep features;
- use scalar invariant summaries for routing/attention decisions;
- mix equivariant feature vectors only with scalar weights.

The encoded features are invariant to the **right crystal group**, not invariant to **left SO(3)**. If they were invariant to left SO(3), the orientation signal would be destroyed.
        """
    ),
    md(
        r"""
## Are we doing projection in the latent space?

There are two different meanings of “projection” here.

### 1. Encoder irrep projection: yes

The local-iso encoder creates orbit-averaged Cartesian tensor features and projects them into irrep channels. This is the quotient/projection step that builds the right-crystal-invariant representation:

$$
T(R)=\frac{1}{|O|}\sum_{u\in O\cdot u_0}(Ru)^{\otimes k},
\qquad
E(q)=T(R(q))P_{\mathrm{irrep}}.
$$

In source this is:

- orbit average: [local_iso_embedding.py:586](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:586), lines 586-595;
- irrep projection: [local_iso_embedding.py:615](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:615), lines 615-617.

### 2. Learned generic latent projection: no

After the encoder, OCRP does **not** repeatedly project arbitrary latent features back into the invariant/equivariant subspace. The preservation is architectural:

| Network component | Is it a generic latent projection? | Why symmetry is preserved |
| --- | --- | --- |
| LR/HR convs | No | They use e3nn tensor products, not arbitrary dense channel mixing: [SR_4x4_from_4x1_ocrp_anchorless.py:353](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:353), lines 353-358, and [SR_4x4_from_4x1_ocrp_anchorless.py:424](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:424), lines 424-430. |
| Clusterer | No | It uses symmetry-aware misorientation on raw quaternions and outputs scalar cluster IDs. |
| Slot builder/router | No | It operates on scalar masks/meta/phase, so ordinary MLPs are allowed there. |
| Cross-attention | No generic irrep-channel projection | It computes scalar weights from invariant summaries, then forms scalar-weighted sums of equivariant `bank_f`. |
| Patch assembly | No | It reshapes/spatially arranges feature tokens; it does not mix irrep channels. |

So the model does not rely on “projecting again later” to fix broken symmetry. If a generic latent projection broke left equivariance, nothing downstream would magically restore it.
        """
    ),
    code(
        r"""
# Counterexample: a generic dense latent projection breaks left SO(3) equivariance.
#
# If Y = X M is to be equivariant with the same output irreps, then for row-vector
# features we need:
#
#   (X A) M = (X M) A
#
# for all X and all rotations, which means A M = M A. A random dense M does not
# commute with the irrep action.

feat_base = base["feat_lr_encode"]
feat_left = left["feat_lr_encode"]

torch.manual_seed(7)
M_bad = torch.randn(model.feature_dim, model.feature_dim, dtype=DTYPE, device=DEVICE)
M_bad = M_bad / math.sqrt(model.feature_dim)

# A symmetry-respecting linear map for one copy of 2e plus one copy of 4e is just
# a scalar gain on each irrep block. This is not the network architecture itself;
# it is a minimal sanity contrast showing what "commutes with D(g)" looks like.
M_block_gain = torch.zeros_like(M_bad)
M_block_gain[:5, :5] = 1.7 * torch.eye(5, dtype=DTYPE, device=DEVICE)
M_block_gain[5:, 5:] = -0.4 * torch.eye(9, dtype=DTYPE, device=DEVICE)

def projection_check(M: torch.Tensor, label: str) -> dict:
    y_base = feat_base @ M
    y_left_actual = feat_left @ M
    y_left_expected = y_base @ A_left_row
    err = tensor_error(y_left_expected, y_left_actual)
    comm = A_left_row @ M - M @ A_left_row
    err.update(
        {
            "latent_map": label,
            "commutator_max_abs": float(comm.abs().max()),
            "commutator_rms": float(comm.square().mean().sqrt()),
        }
    )
    return err

projection_rows = [
    projection_check(M_bad, "generic dense 14x14 map: breaks equivariance"),
    projection_check(M_block_gain, "per-irrep scalar gains: preserves equivariance"),
]
projection_df = pd.DataFrame(projection_rows)
projection_csv = OUT_DIR / "latent_projection_counterexample.csv"
projection_df.to_csv(projection_csv, index=False)
print("saved:", projection_csv)
display(
    projection_df[
        [
            "latent_map",
            "rel",
            "rms",
            "max_abs",
            "commutator_max_abs",
            "commutator_rms",
        ]
    ]
)
        """
    ),
    md(
        r"""
## Stage 1 — Local-iso encoder

Source path:

- `model.encode(...)` dispatch: [SR_4x4_from_4x1_ocrp_anchorless.py:2016](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2016), lines 2016-2019.
- Full encoder wrapper: [SR_4x4_from_4x1_ocrp_anchorless.py:490](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:490), lines 490-492.
- Passive API input is conjugated to active: [local_iso_embedding.py:647](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:647).
- Active rotation matrix is built: [local_iso_embedding.py:648](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:648).
- Orbit averaging and irrep projection happen at [local_iso_embedding.py:586](/data/home/umang/Materials/Reynolds-QSR_paper/models/local_iso_embedding.py:586), lines 586-617.

Why this stage has the target symmetry:

- right \(q\otimes h\), \(h\in O\): \(h\) permutes the cubic orbit before averaging, so the average is unchanged;
- left \(g\otimes q\): the active tensor is physically rotated in lab space, so its projected irrep channels transform by \(D(g)\).
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["feat_lr_encode"])])
        """
    ),
    md(
        r"""
## Stage 2 — LR cosine-masked equivariant convolution

Source path:

- LR conv is called in `_forward_sr_features` at [SR_4x4_from_4x1_ocrp_anchorless.py:2038](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2038), lines 2038-2040.
- Convolution implementation is [SR_4x4_from_4x1_ocrp_anchorless.py:315](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:315), lines 315-441.
- Cosine mask weights are computed at [SR_4x4_from_4x1_ocrp_anchorless.py:390](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:390), lines 390-402.
- Neighbor average and tensor product happen at [SR_4x4_from_4x1_ocrp_anchorless.py:424](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:424), lines 424-430.

Why this stage preserves the target symmetry:

- the cosine mask is built from feature dot products/norms, so it is invariant under orthogonal \(D(g)\);
- the weighted neighbor feature is an equivariant feature;
- the e3nn tensor product maps equivariant inputs to equivariant outputs;
- right crystal invariance has already been quotient-projected by the encoder, so the conv sees the same input features under right crystal representative changes.
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["feat_lr_pre_ocrp"])])
        """
    ),
    md(
        r"""
## Stage 3 — OCRP support banks and quaternion clustering

Source path:

- OCRP is called at [SR_4x4_from_4x1_ocrp_anchorless.py:2042](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2042), lines 2042-2047.
- Support banks are built at [SR_4x4_from_4x1_ocrp_anchorless.py:1492](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1492), lines 1492-1505.
- The local patch-bank helper is [SR_4x4_from_4x1_ocrp_anchorless.py:174](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:174), lines 174-198.
- Clusterer forward is [SR_4x4_from_4x1_ocrp_anchorless.py:800](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:800), lines 800-849.
- Symmetry-aware misorientation edges are computed at [SR_4x4_from_4x1_ocrp_anchorless.py:817](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:817), lines 817-819.

Important distinction:

- `bank_f` is an irrep-valued feature bank, so it must be right-invariant and left-equivariant.
- `bank_q` is a raw quaternion support carrier. It is **not** itself a quotient feature tensor. Under active left \(g\), the active form of `bank_q` transforms as \(g\otimes q\). Under active right crystal representative changes, raw quaternions change, but the clusterer uses symmetry-aware misorientation, so `cluster_ids` remain invariant.
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["bank_f", "cluster_ids"])])

bank_q_base_active = checkpoint_output_to_active(base["bank_q"])
bank_q_right_active = checkpoint_output_to_active(right["bank_q"])
bank_q_left_active = checkpoint_output_to_active(left["bank_q"])
bank_q_left_expected = left_so3_action_active(
    g_left.view(1, 1, 1, 4).expand_as(bank_q_base_active),
    bank_q_base_active,
)
raw_bank_rows = [
    {
        "raw_bank_q_check": "right crystal changes raw representatives",
        **tensor_error(bank_q_base_active, bank_q_right_active),
    },
    {
        "raw_bank_q_check": "left SO3 transforms raw active bank as g⊗q",
        **tensor_error(bank_q_left_expected, bank_q_left_active),
    },
]
display(pd.DataFrame(raw_bank_rows))
        """
    ),
    md(
        r"""
## Stage 4 — Slots and router

Source path:

- Slot builder starts at [SR_4x4_from_4x1_ocrp_anchorless.py:852](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:852).
- Slot masks and metadata are emitted at [SR_4x4_from_4x1_ocrp_anchorless.py:960](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:960), lines 960-1008.
- Patch router starts at [SR_4x4_from_4x1_ocrp_anchorless.py:1037](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1037).
- Router forward maps scalar masks and phase embeddings to logits at [SR_4x4_from_4x1_ocrp_anchorless.py:1094](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1094), lines 1094-1166.

These tensors are routing/index/scalar quantities. They do not carry \(l=2\) or \(l=4\) orientation channels. Therefore they should be invariant under:

- right crystal representative changes;
- common left SO(3) frame rotation.

They can be scalar-invariant because they only decide which equivariant features get mixed. Scalar weights times equivariant feature vectors preserve equivariance.
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["slot_mask", "slot_valid", "slot_meta", "router_logits", "owner_idx"])])
        """
    ),
    md(
        r"""
## Stage 5 — Anchorless member cross-attention proposals and patch assembly

Source path:

- Cross-attention forward starts at [SR_4x4_from_4x1_ocrp_anchorless.py:1225](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1225).
- Invariant member summaries are norms of irrep blocks at [SR_4x4_from_4x1_ocrp_anchorless.py:1011](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1011), lines 1011-1034.
- Keys/queries and masked scalar attention weights are built at [SR_4x4_from_4x1_ocrp_anchorless.py:1245](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1245), lines 1245-1320.
- Proposed patch features are weighted sums of `bank_f` at [SR_4x4_from_4x1_ocrp_anchorless.py:1321](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1321), lines 1321-1323.
- Hard owner selection and patch assembly occur at [SR_4x4_from_4x1_ocrp_anchorless.py:1623](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:1623), lines 1623-1653.

Why equivariance survives:

- attention logits/weights are scalar;
- `patch_prop` is a scalar-weighted sum of equivariant `bank_f`;
- `patch_out` selects/mixes equivariant proposals using scalar owner masks;
- assembling patch tokens into the HR grid only changes spatial layout, not orientation channels.

If `xattn_alpha` is not exactly identical between branches, that is a diagnostic softmax-level numerical effect. The proof target is the equivariant feature output after applying those weights, and that is checked directly by `patch_prop`, `patch_out`, and `feat_hr_raw_ocrp`.
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["xattn_alpha", "patch_prop", "patch_out", "feat_hr_raw_ocrp"])])
        """
    ),
    md(
        r"""
## Stage 6 — HR equivariant refinement convolutions

Source path:

- HR conv1 is applied at [SR_4x4_from_4x1_ocrp_anchorless.py:2054](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2054), lines 2054-2056.
- HR conv2 is applied at [SR_4x4_from_4x1_ocrp_anchorless.py:2057](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2057), lines 2057-2060.
- Probe tensors are inserted into `aux` at [SR_4x4_from_4x1_ocrp_anchorless.py:2067](/data/home/umang/Materials/Reynolds-QSR_paper/models/SR_4x4_from_4x1_ocrp_anchorless.py:2067), lines 2067-2086.

The HR convs are the same class as the LR conv. Their mask is scalar and their tensor product is equivariant, so they preserve:

- right crystal invariance;
- left SO(3) equivariance.
        """
    ),
    code(
        r"""
display(df[df["tensor"].isin(["feat_hr_post_hr_conv1", "feat_hr_post_hr_conv2", "feat_hr_final"])])
        """
    ),
    md(
        r"""
## Explicit “not vice versa” controls

Now we check the common mistakes directly.

The reason the opposite cannot be correct is structural. In active convention, left and right multiplication act on different sides of the rotation matrix:

$$
R(g\otimes q)=R(g)R(q),
\qquad
R(q\otimes h)=R(q)R(h).
$$

The left factor \(R(g)\) rotates the **lab-frame tensor indices**. The right factor \(R(h)\), when \(h\in O\), only relabels the **crystal-frame representative** that is averaged/quotiented out. Swapping the two claims would mix up these two indices.

### Wrong idea 1: “left SO(3) should be invariant”

False for non-scalar irrep features. A left global rotation physically rotates the tensor; the features should change by \(D(g)\). Therefore:

$$
X(g\otimes q) \neq X(q)
$$

in general, while:

$$
X(g\otimes q) \approx X(q)A_{\mathrm{row}}(g).
$$

If left SO(3) were invariant for every \(g\), then every physical lab-frame rotation of the same crystal would have the same feature:

$$
X(g\otimes q)=X(q),\qquad \forall g\in SO(3).
$$

That would collapse the orientation signal itself. Non-scalar \(l=2\) and \(l=4\) channels could not carry lab-frame directional information.

### Wrong idea 2: “right crystal action should be equivariant by \(D(h)\)”

False. Right multiplication by a crystal symmetry changes the representative inside the same quotient class, so:

$$
X(q\otimes h)\approx X(q),
$$

not:

$$
X(q\otimes h)\approx X(q)A_{\mathrm{row}}(h).
$$

That wrong equation treats a crystal-basis relabeling as if it were a physical left rotation of the sample frame. It is the wrong group action for active orientations. The correct right action disappears inside the Reynolds/orbit average; the correct left action remains visible as a channel rotation.

### Wrong idea 3: “right multiplication by any SO(3) rotation should be invariant”

False. Right-invariance is only for the crystal group \(O\), not arbitrary SO(3):

$$
X(q\otimes r)\not\approx X(q),\qquad r\notin O.
$$

If arbitrary right SO(3) rotations were quotiented out, the encoder would average over the full sphere rather than the finite cubic orbit, killing the non-scalar anisotropic information that the OCRP model is explicitly trained to use.
        """
    ),
    code(
        r"""
# Fixed right crystal symmetry, used for wrong-side D(h) comparison.
h_fixed = sym_ops_active[3]
lr_crop_right_fixed_active = right_crystal_action_active(
    lr_crop_active,
    h_fixed.view(1, 1, 4).expand_as(lr_crop_active),
)
right_fixed, _ = run_feature_pipeline_from_active(lr_crop_right_fixed_active)

# Generic right SO(3) rotation that is not a cubic symmetry.
r_right_generic = qnormalize_torch(torch.tensor([0.801, 0.211, -0.413, 0.376], dtype=DTYPE, device=DEVICE).view(1, 4))[0]
lr_crop_right_generic_active = right_crystal_action_active(
    lr_crop_active,
    r_right_generic.view(1, 1, 4).expand_as(lr_crop_active),
)
right_generic, _ = run_feature_pipeline_from_active(lr_crop_right_generic_active)

A_h_row = Irreps(model.irreps_feat).D_from_matrix(
    quat_to_matrix_active_torch(h_fixed).detach().cpu()
).to(device=DEVICE, dtype=DTYPE).T.contiguous()

control_rows = []
for name in ["feat_lr_encode", "feat_hr_final"]:
    control_rows.extend(
        [
            {
                "tensor": name,
                "control": "correct: right crystal h is invariant",
                **tensor_error(base[name], right_fixed[name]),
            },
            {
                "tensor": name,
                "control": "wrong: right crystal h treated as left equivariant D(h)",
                **tensor_error(apply_feature_action_row(base[name], A_h_row), right_fixed[name]),
            },
            {
                "tensor": name,
                "control": "correct: left SO3 g is equivariant",
                **tensor_error(apply_feature_action_row(base[name], A_left_row), left[name]),
            },
            {
                "tensor": name,
                "control": "wrong: left SO3 g treated as invariant",
                **tensor_error(base[name], left[name]),
            },
            {
                "tensor": name,
                "control": "wrong: right generic SO3 r treated as crystal-invariant",
                **tensor_error(base[name], right_generic[name]),
            },
        ]
    )

controls = pd.DataFrame(control_rows)
control_csv = OUT_DIR / "not_vice_versa_controls.csv"
controls.to_csv(control_csv, index=False)
print("saved:", control_csv)
display(controls[["tensor", "control", "rel", "rms", "max_abs"]])
        """
    ),
    md(
        r"""
## Optional visualization of the proof errors

The plot below is diagnostic only. The proof is the layerwise table above.

Expected pattern:

- right crystal error: near numerical precision;
- left equivariance error: near numerical precision;
- wrong left-as-invariant error: order-one for non-scalar irrep features.
        """
    ),
    code(
        r"""
import os

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "mplconfig"))
(OUT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

plot_df = df[df["tensor_type"] == "irrep feature"].copy()
plot_df["right_log10"] = np.log10(plot_df["right_rel"].clip(lower=1e-12))
plot_df["left_eq_log10"] = np.log10(plot_df["left_eq_rel"].clip(lower=1e-12))
plot_df["wrong_left_inv_log10"] = np.log10(plot_df["wrong_left_as_invariant_rel"].clip(lower=1e-12))

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(plot_df))
width = 0.25
ax.bar(x - width, plot_df["right_log10"], width, label="right crystal invariant error")
ax.bar(x, plot_df["left_eq_log10"], width, label="left SO3 equivariant error")
ax.bar(x + width, plot_df["wrong_left_inv_log10"], width, label="wrong: left as invariant")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["tensor"], rotation=45, ha="right")
ax.set_ylabel("log10(relative error)")
ax.set_title("Active-convention layerwise symmetry checks")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
plot_path = OUT_DIR / "layerwise_symmetry_errors.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
print("saved:", plot_path)
        """
    ),
    md(
        r"""
## Final conclusion

For the active-convention OCRP feature pipeline:

1. The local-iso encoder makes the crystal quotient on the **right**:

   $$
   q \sim q\otimes h,\qquad h\in O.
   $$

2. Every irrep-valued feature tensor after that remains:

   $$
   \text{right-crystal invariant}
   \quad\text{and}\quad
   \text{left-SO(3) equivariant}.
   $$

3. Scalar OCRP routing/index tensors are type-0 quantities. The hard routing objects such as `cluster_ids`, `slot_mask`, `slot_meta`, `router_logits`, and `owner_idx` are invariant in the table. The returned `xattn_alpha` diagnostic is scalar too, but shows tiny softmax-level numerical differences; the actual scalar-weighted feature outputs still satisfy the symmetry checks.

4. The side still matters after encoding. Right crystal invariance is inherited by feature-only blocks, but left SO(3) equivariance must be preserved by every operation that touches irrep channels. A generic dense latent projection breaks it; the notebook's counterexample shows this explicitly.

5. There is no magic repeated latent-space projection that fixes symmetry after arbitrary channel mixing. The real projection is the encoder irrep projection; later layers preserve symmetry by construction through equivariant tensor products, scalar invariant routing quantities, and scalar-weighted sums.

6. The wrong-side controls show that this is **not vice versa**:

   - left SO(3) is not an invariance for non-scalar features;
   - right crystal action is not treated as a left \(D(h)\)-equivariance;
   - arbitrary right SO(3) rotations are not crystal invariances.

This is exactly the active-convention behavior:

$$
\boxed{
F(q\otimes h)=F(q),\ h\in O
}
\qquad
\boxed{
F(g\otimes q)=F(q)D(g)^\mathsf{T},\ g\in SO(3)
}
$$

with the transpose appearing only because this implementation stores feature channels as row vectors.
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (material)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT} with {len(cells)} cells")
