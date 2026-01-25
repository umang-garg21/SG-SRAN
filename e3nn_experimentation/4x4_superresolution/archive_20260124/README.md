# Archive: Original Code from outputs_20260124_163036

This folder contains the **original version** of the training code that produced
the outputs in `outputs_20260124_163036/`.

## Key Differences from Current Version

### train_e3nn_SRnet_original.py
- Uses **HR subsampling** to create LR input (not separate LR_Data folder)
- Uses **2-row visualization** (Input vs Output format)
- Functions: `render_input_output_comparison()` instead of `render_lr_hr_sr_comparison()`
- train_model takes single `train_data` parameter (HR only)

### Current version (train_e3nn_SRnet.py)
- Loads from separate **LR_Data** and **HR_Data** folders
- Uses **3-row visualization** (LR/HR/SR on same image)
- LR displayed at native resolution without bilinear upsampling

## Files

| File | Description |
|------|-------------|
| train_e3nn_SRnet_original.py | Original training script (HR subsampling, 2-row viz) |
| e3nn_SRnet.py | Network architecture (same as current - includes bilinear skip) |

## Output Files from outputs_20260124_163036/
- ipf_comparison_strict.png
- ipf_comparison_nonstrict.png
- ipf_ground_truth_strict.png
- ipf_ground_truth_nonstrict.png
- model_strict_equivariance.pt (84KB)
- model_non_strict_equivariance.pt (200KB)

## To Run This Version
```bash
cd /data/home/umang/Materials/e3nn_Reynolds/e3nn_experimentation/4x4_superresolution/archive_20260124
python train_e3nn_SRnet_original.py
```

Note: This version imports `e3nn_SRnet` from the same folder, which has the
bilinear skip connection in the network forward pass.

## Date
Original outputs generated: January 24, 2026 at 16:30:36
Archive created: January 25, 2026
