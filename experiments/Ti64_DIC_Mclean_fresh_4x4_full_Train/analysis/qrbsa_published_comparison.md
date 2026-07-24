# Published QRBSA Comparison

This note compares the current Ti64 DIC McLean x4 full-test results against the metrics reported for the published QRBSA paper. The only directly overlapping quantitative metrics are IPF-map PSNR and SSIM, plus model parameter count. The published paper does not report the current orientation-error percentiles, boundary F1, tolerance-aware boundary metrics, grain-count/size metrics, or nearest-neighbor distance.

## Sources

- Published QRBSA paper: Jangid et al., "Q-RBSA: High-Resolution 3D EBSD Map Generation Using An Efficient Quaternion Transformer Network", arXiv:2303.10722.
- Published numbers extracted from the arXiv TeX source, table `tab:psnr_ssim_network_x4`.
- Local numbers from `metrics.json` in this folder.
- Local Q-RBSA config: `qrbsa_adapted_4x4_fresh_ti64_dic_mclean/config.json`.

## Protocol Difference

The published QRBSA experiment is a sparse-z 3D EBSD reconstruction task: LR data are made by removing xy planes along z, and QRBSA predicts missing rows/planes from xz or yz 2D slices. The paper reports IPF-map PSNR/SSIM for Ti-6Al-4V, Ti-7Al 1%, and Ti-7Al 3% at x2 and x4.

The current experiment is `Ti64_DIC_Mclean_QSR_x4`, using the existing split with 581 train, 73 validation, and 72 test samples. The local Q-RBSA entry is an adapted 2D isotropic x4 run retaining the published Q-RBSA head/body but using a local data adapter, D6 loss from D6h metadata, and an isotropic 2D quaternion pixel-shuffle tail. Therefore the absolute PSNR/SSIM values are not a strict reproduction of the published QRBSA table.

## Published QRBSA x4 Results

| Network | Params | Ti-6Al-4V PSNR | Ti-6Al-4V SSIM | Ti-7Al 1% PSNR | Ti-7Al 1% SSIM | Ti-7Al 3% PSNR | Ti-7Al 3% SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAN | 63,315,578 | 17.55 | 0.673 | 26.54 | 0.849 | 28.90 | 0.905 |
| EDSR | 6,355,460 | 18.09 | 0.718 | 27.34 | 0.907 | 29.57 | 0.932 |
| QEDSR | 1,593,092 | 18.04 | 0.710 | 27.22 | 0.904 | 29.52 | 0.930 |
| QRBSA | 5,952,782 | 18.20 | 0.730 | 27.48 | 0.908 | 29.65 | 0.940 |

In the published table, QRBSA is best on both PSNR and SSIM for all three x4 titanium datasets.

## Current Local x4 Results

| Method | Params | PSNR IPF-XYZ | SSIM IPF-XYZ | Mean misorientation deg | Boundary F1 | Boundary F1 tol1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EDSR | 6,355,460 | 14.369 | 0.552 | 10.041 | 0.642 | 0.957 |
| Q-RBSA | 6,174,734 | 14.285 | 0.557 | 10.413 | 0.621 | 0.956 |
| QEDSR | 1,815,044 | 14.364 | 0.560 | 10.089 | 0.622 | 0.957 |
| OCRP w5-local | n/a | 14.653 | 0.579 | 9.438 | 0.636 | 0.944 |

Only PSNR and SSIM are matched to the published QRBSA paper. The orientation, boundary, and grain metrics are local-only additions.

## Closest Numerical Comparison

Using published Ti-6Al-4V x4 QRBSA as the closest material/scale reference:

| Comparison | PSNR | SSIM | Params |
| --- | ---: | ---: | ---: |
| Published QRBSA Ti-6Al-4V x4 | 18.200 | 0.730 | 5,952,782 |
| Local Q-RBSA Ti64 DIC McLean x4 | 14.285 | 0.557 | 6,174,734 |
| Local - published | -3.915 | -0.173 | +221,952 (+3.73%) |

## Interpretation

The local Q-RBSA run does not reproduce the published QRBSA ranking. In the published x4 table, QRBSA is above EDSR and QEDSR on PSNR/SSIM. In the current Ti64 DIC McLean x4 table, Q-RBSA is slightly below EDSR/QEDSR in PSNR and slightly below QEDSR in SSIM, while OCRP w5-local is the best local method on PSNR, SSIM, and mean orientation error.

The most likely reason is not simply model implementation quality, but protocol shift: the published task is 1D sparse-z plane recovery in 3D EBSD volumes, while the current run is an adapted isotropic 2D x4 setting on a different Ti64 DIC McLean split. The local task also exposes stronger boundary and small-grain penalties, where Q-RBSA does not dominate. Thus the fair paper-facing statement is: published QRBSA is strong on its reported sparse-z IPF PSNR/SSIM benchmark, but under the current isotropic Ti64 DIC McLean x4 benchmark its adapted Q-RBSA implementation is not the strongest baseline, and OCRP w5-local is better on the matching image metrics and on the orientation-aware metrics we added.
