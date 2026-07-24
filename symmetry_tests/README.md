# Symmetry Block Tests

This folder contains focused symmetry tests for each major SR model block.
Each file is intentionally isolated so the checks are easy to read and debug.

## Files

- `test_encoder_block.py`: encoder right-invariance under crystal group `G`, plus left-equivariance for `G` and `SO(3)`.
- `test_equivariant_conv_block.py`: spatial equivariant-conv block checks for right/left symmetry behavior.
- `test_upsampler_block.py`: transpose-conv upsampler checks for right/left symmetry behavior.
- `test_attention_block.py`: block-local attention checks for right/left symmetry behavior.
- `test_decoder_block.py`: decoder checks for right-invariance and left-equivariance (for `G` and `SO(3)`).
- `_symmetry_utils.py`: shared quaternion math, group-action helpers, and error metrics.

## Run

```bash
pytest symmetry_tests -q
```
