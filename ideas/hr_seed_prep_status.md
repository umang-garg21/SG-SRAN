# HR Seed Prep: Condensed Implementation Status

This note summarizes where the `ideas/hr_seed_prep.ipynb` exploration currently stands.

## Goal

We are exploring HR seed construction in the notebook first, before integrating any validated idea into [models/irrep_sr_true_attention.py](/data/home/umang/Materials/Reynolds-QSR_4x1/models/irrep_sr_true_attention.py).

The working idea is:

1. Encode LR input into `a1` invariant features.
2. Build translated LR pairings.
3. Apply `o3.TensorProduct` / full TP between `LR` and shifted `LR`.
4. Extract the `4e` block so the current decoder can consume it directly.
5. Decode and visualize the result.
6. Use selected offset-specific decoded feature maps to construct an HR seed.

## Environment

- Use conda environment: `material`
- The notebook is the main playground: [ideas/hr_seed_prep.ipynb](/data/home/umang/Materials/Reynolds-QSR_4x1/ideas/hr_seed_prep.ipynb)

## What Is Implemented In The Notebook

- Model + sample loading for the current `irrep_sr_true_attention` setup.
- LR feature extraction via the model encoder.
- TP feature construction between LR features and shifted LR features.
- Offset sweep over a small bank of LR translations.
- Extraction of the decoder-compatible `4e` TP block.
- Decoding of the produced features.
- IPF visualization of decoded outputs.
- Utilities for comparing decoded outputs across offsets.

## Current Offset Bank

The notebook currently explores this LR offset bank:

`[(0, 0), (-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1), (1, 1), (-1, 1), (1, -1)]`

This is the broader exploration set used to inspect how TP changes under neighbor direction.

## Current HR Seed Construction

The notebook no longer does mask-weighted averaging for the HR seed.

Instead, it now uses a stitched construction:

- Selected offsets:
  - `(-1, 0)`
  - `(0, 1)`
  - `(1, 0)`
  - `(0, 0)`
- These four LR TP feature maps are treated as four row phases.
- For each LR row, the four selected maps are stacked in a fixed order to form four HR rows.
- This produces a stitched feature map of shape `4H x W` if the LR map is `H x W`.
- That stitched feature map is then passed through the decoder.
- The notebook visualizes the constructed HR seed after decoding.

In short: we are currently forming the HR seed by phase-wise stitching, not blending.

## Important Interpretation

This stitched version is useful as an experiment, but it is still heuristic.

- It gives exactly 4 phase maps, which matches `4x` height upsampling.
- It is not yet a principled geometric derivation of the 4 HR subrows.
- The chosen offsets are enough to visualize whether directional TP features contain useful seed information.
- They are not yet sufficient evidence for the final model design.

## Main Open Question

The key question now is not whether the notebook can produce an HR seed. It can.

The real question is whether the selected stitched offsets produce:

- meaningful structure,
- stable phase behavior,
- and no obvious 4-row striping or seam artifacts.

## Intended Next Step

Use the notebook to compare stitched HR seeds from different offset selections and judge them visually before moving any of this logic into the final model file.
