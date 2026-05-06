# OCRP Stage Probe

- Experiment: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01`
- Config: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/logs/run_config.json`
- Checkpoint: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/checkpoints/best_model.pt`
- Split / sample: `Val` / `0`
- Device: `cpu`
- Model: `models.SR_ocrp.IsoEmbeddingSROCRP`
- OCRP mode: `pixel_patch`
- Anchor builder: `MedoidSlotContextBuilder`
- Support grid: `32x32`
- HR shape: `128x128`
- Active slots: `[0, 1, 2]`

## Slot Usage

| slot | support coverage | HR owner usage |
| --- | ---: | ---: |
| 0 | 1.0000 | 0.9565 |
| 1 | 0.4395 | 0.0381 |
| 2 | 0.1465 | 0.0054 |

## Output Files

- decoded_main_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/ocrp_stage_probe/val_sample0000/decoded_main_gallery.png`
- decoded_support_context_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/ocrp_stage_probe/val_sample0000/decoded_support_context_gallery.png`
- decoded_slot_proposal_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/ocrp_stage_probe/val_sample0000/decoded_slot_proposal_gallery.png`
- stage_metrics_csv: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/ocrp_stage_probe/val_sample0000/stage_metrics.csv`
- probe_bundle: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/ocrp_stage_probe/val_sample0000/probe_bundle.pt`

## Stage Metrics

| stage | shape | mean deg | p95 deg | max deg | note |
| --- | --- | ---: | ---: | ---: | --- |
| lr_input | 32x32 | 2.801 | 21.391 | 59.819 | Input LR quaternion field. |
| encode_lr | 32x32 | 2.829 | 21.492 | 59.924 | Encoder output decoded back on the LR grid. |
| lr_conv1_pre_ocrp | 32x32 | 2.792 | 21.530 | 59.919 | LR spatial context just before OCRP support-bank construction. |
| selected_patch_out_hr | 128x128 | 1.669 | 1.412 | 60.245 | Hard-routed OCRP output before the HR post-conv stack. |
| hr_conv1_post_ocrp | 128x128 | 1.668 | 1.408 | 60.206 | HR conv 1 output. |
| hr_conv2_post_ocrp | 128x128 | 1.661 | 1.403 | 60.218 | HR conv 2 output. |
| support_slot0_anchor_ctx | 32x32 | 2.980 | 22.117 | 59.875 | Current slot anchor feature decoded on the support grid. |
| support_slot1_anchor_ctx | 32x32 | 37.767 | 59.768 | 60.195 | Current slot anchor feature decoded on the support grid. |
| support_slot2_anchor_ctx | 32x32 | 36.202 | 53.201 | 60.166 | Current slot anchor feature decoded on the support grid. |
| support_slot0_pooled_mean | 32x32 | 2.903 | 21.926 | 59.923 | Token-wise within-slot pooled context, averaged over HR patch tokens. |
| support_slot1_pooled_mean | 32x32 | 37.751 | 59.778 | 60.009 | Token-wise within-slot pooled context, averaged over HR patch tokens. |
| support_slot2_pooled_mean | 32x32 | 36.194 | 53.176 | 60.113 | Token-wise within-slot pooled context, averaged over HR patch tokens. |
| slot0_proposal_hr | 128x128 | 2.585 | 1.718 | 60.620 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot1_proposal_hr | 128x128 | 44.185 | 59.718 | 60.679 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot2_proposal_hr | 128x128 | 45.660 | 58.674 | 61.402 | Decoded HR patch proposal emitted by this slot before routing selection. |
| sr_output | 128x128 | 1.661 | 1.403 | 60.218 | Final decoded SR prediction. |
| hr_target | 128x128 | 0.000 | 0.000 | 0.000 | Ground-truth HR quaternion field. |
