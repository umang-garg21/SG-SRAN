# Cubic Invariance Notes for e3nn Quaternion Encoders

This project uses an invariant-subspace Wigner construction to encode orientation quaternions into cubic-invariant features.

## Conventions

- Quaternion layout: scalar-first `wxyz`.
- Input convention for Bunge paths: passive crystal orientation quaternions.
- Internal convention for e3nn Wigner computation: active rotation, obtained by quaternion conjugation.
- Symmetry group for SO(3) invariance: proper cubic group `O` (24 rotations).
  - Full cubic Laue group `Oh` (`m-3m`) has 48 elements (includes improper operations).

## Invariant Basis Construction

For each degree `l`, construct the Reynolds projector:

`P_l = (1 / |G|) * sum_{g in G} D^l(g)`

where `D^l(g)` is the Wigner-D representation for group element `g`.

The cubic-invariant subspace is the eigenspace of `P_l` with eigenvalue near `1`.
Its basis is stored as `U_l`.

Encoder block for degree `l`:

`F_l(q) = D^l(q) U_l`

Flatten and concatenate over selected degrees to produce the invariant feature vector.

## Expected Cubic-Invariant Ranks (Proper Cubic Group, 24 Elements)

These are the expected `rank(U_l)` values used by the model and tests:

| Degree `l` | `rank(U_l)` |
| --- | --- |
| 4 | 1 |
| 6 | 1 |
| 8 | 1 |
| 10 | 1 |
| 12 | 2 |

These values are checked in `tests/test_e3nn_invariant_ranks.py` to catch convention or symmetry-table drift.

## Symmetry-Error Metrics (LaTeX)

Let `E(q)` be the invariant encoder output for quaternion `q`, `G` the proper cubic group (`|G|=24`), and `{q_i}_{i=1}^N` sampled unit quaternions.

### 1) Invariance residual

\[
\varepsilon_{\mathrm{inv}}
=
\frac{1}{N|G|}
\sum_{i=1}^{N}
\sum_{s\in G}
\left\|
E(q_i)-E(s^{-1}\otimes q_i)
\right\|_2^2
\]

### 2) Nearest-neighbor symmetry-aware misorientation

Define nearest neighbor in feature space:
\[
j^\*(i)=\arg\min_{j\neq i}\|E(q_i)-E(q_j)\|_2.
\]

Define orbit-reduced angular distance:
\[
d_G(q_a,q_b)=
\min_{s\in G}
2\arccos\!\left(
\left|\left\langle q_a,\;s^{-1}\otimes q_b\right\rangle\right|
\right).
\]

Then the per-sample symmetry error is
\[
\theta_i=d_G\!\left(q_i,\;q_{j^\*(i)}\right),
\]
and summary statistics use mean/median/p95 of `\{\theta_i\}`.

### 3) Collision rate above tolerance

For angular tolerance `\tau` (degrees):
\[
\mathrm{CollisionRate}_\tau
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[\theta_i>\tau\right].
\]

### Gate-based L-set adequacy

For a candidate degree set `L`, accept if:
\[
\varepsilon_{\mathrm{inv}}(L)\le\epsilon_{\mathrm{inv}},
\quad
\mathrm{p95}(\theta(L))\le\epsilon_{\mathrm{SE}},
\quad
\mathrm{CollisionRate}_\tau(L)\le\epsilon_{\mathrm{col}}.
\]
