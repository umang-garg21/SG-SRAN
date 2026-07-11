# Overall Metric Winners With Classical Baselines

This summary uses the combined 12-method table in `metrics.md`: `Nearest`, `Bicubic`, `SLERP`, `Symm-SLERP`, `Atindama`, `EDSR`, `HAN`, `RCAN`, `SAN`, `Q-RBSA`, `QEDSR`, and `OCRP w5-local`.

## Winner Count

| method | metric wins |
| --- | ---: |
| OCRP w5-local | 11 |
| Symm-SLERP | 2 |
| Atindama | 2 |
| RCAN | 1 |
| Nearest | 1 |
| EDSR | 1 |
| SAN | 1 |

## Per-Metric Winners

| metric | direction | winner | value |
| --- | --- | --- | ---: |
| mean_deg | lower | OCRP w5-local | 9.437733 |
| median_deg | lower | OCRP w5-local | 0.479697 |
| p90_deg | lower | RCAN | 34.804958 |
| p95_deg | lower | Symm-SLERP | 56.123993 |
| p99_deg | lower | Symm-SLERP | 79.567749 |
| tol1_mean_deg | lower | OCRP w5-local | 2.151019 |
| interior_mean_deg | lower | OCRP w5-local | 0.516229 |
| boundary_band_mean_deg | lower | OCRP w5-local | 9.525213 |
| boundary_precision | higher | OCRP w5-local | 0.700612 |
| boundary_recall | higher | Atindama | 0.952832 |
| boundary_f1 | higher | EDSR | 0.642265 |
| boundary_precision_tol1 | higher | OCRP w5-local | 0.985656 |
| boundary_recall_tol1 | higher | Atindama | 0.997071 |
| boundary_f1_tol1 | higher | SAN | 0.961752 |
| grain_log10_size_wasserstein | lower | OCRP w5-local | 0.167483 |
| distance_to_nn_deg | lower | Nearest | 0.011451 |
| grain_ratio_abs_log10 | lower | OCRP w5-local | 0.057663 |
| psnr_ipf_xyz_db | higher | OCRP w5-local | 14.653028 |
| ssim_ipf_xyz | higher | OCRP w5-local | 0.579120 |

## Interpretation

`OCRP w5-local` remains the best overall method after adding the non-learnable baselines. It wins the main orientation-fidelity metrics, the interior and boundary-band orientation errors, boundary precision, grain-distribution metrics, and both IPF image-fidelity metrics.

The non-learnable baselines are still useful context, but they do not change the overall ranking. `Symm-SLERP` has the best p95/p99 tail errors, and `Nearest` is trivially closest to nearest-neighbor because it is the nearest-neighbor upsample itself. Those wins should not be read as broad superiority. `Atindama` wins recall-like boundary metrics by producing many boundary positives, while `EDSR` and `SAN` win the two F1-style boundary metrics.
