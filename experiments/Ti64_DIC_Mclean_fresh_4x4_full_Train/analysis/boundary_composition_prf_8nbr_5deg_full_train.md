# Boundary Spurious Composition

Protocol: HR boundary centers use an 8-neighbour, symmetry-aware 5 degree contrast rule. For each center, the clipped HR 3x3 window defines the allowed GT orientation composition; each SR pixel in the corresponding clipped 3x3 window is spurious if it is farther than 5 degrees from every HR orientation in that original GT window. Recall is group-level: HR-window orientations are clustered by the transitive closure of the 5 degree adjacency, and a GT group is recovered if any SR-window pixel matches any member of that group. Overlapping GT boundary windows are counted as separate observations.

## Ti64 DIC McLean fresh 4x4 (D6h)

| method | valid | obs. | spurious | FP rate | precision | GT groups | recovered | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAN | 72 | 5952542 | 628578 | 0.105598 | 0.894402 | 1560622 | 1140405 | 0.730737 | 0.804328 |
| OCRP w5-local | 72 | 5952542 | 553232 | 0.092940 | 0.907060 | 1560622 | 1071446 | 0.686551 | 0.781549 |
| Q-RBSA | 72 | 5952542 | 450461 | 0.075675 | 0.924325 | 1560622 | 1036266 | 0.664008 | 0.772834 |
| QEDSR | 72 | 5952542 | 616523 | 0.103573 | 0.896427 | 1560622 | 1044019 | 0.668976 | 0.766177 |
| EDSR | 72 | 5952542 | 875895 | 0.147146 | 0.852854 | 1560622 | 1017925 | 0.652256 | 0.739187 |
| Nearest | 72 | 5952542 | 889207 | 0.149383 | 0.850617 | 1560622 | 904244 | 0.579413 | 0.689298 |
| HAN | 72 | 5952542 | 1660960 | 0.279034 | 0.720966 | 1560622 | 978799 | 0.627185 | 0.670814 |
| RCAN | 72 | 5952542 | 2832828 | 0.475902 | 0.524098 | 1560622 | 779301 | 0.499353 | 0.511426 |
| Symm-SLERP | 72 | 5952542 | 3372032 | 0.566486 | 0.433514 | 1560622 | 596288 | 0.382084 | 0.406177 |
| SLERP | 72 | 5952542 | 4039995 | 0.678701 | 0.321299 | 1560622 | 470076 | 0.301211 | 0.310931 |
| Bicubic | 72 | 5952542 | 4369936 | 0.734129 | 0.265871 | 1560622 | 510270 | 0.326966 | 0.293270 |
| Atindama | 72 | 5952542 | 4711834 | 0.791567 | 0.208433 | 1560622 | 558786 | 0.358053 | 0.263485 |
