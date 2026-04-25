# OCRP Macro Stage Probe

- Experiment: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01`
- Config: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/logs/run_config.json`
- Checkpoint: `experiments/IN718/iso_embedding_ocrp_macro_01/checkpoints/best_model.pt`
- Split / sample: `Val` / `0`
- Device: `cpu`
- Support grid: `32x32`
- HR shape: `128x128`
- Active slots: `[0, 1, 2]`

## Slot Usage

| slot | support coverage | HR owner usage |
| --- | ---: | ---: |
| 0 | 1.0000 | 0.9585 |
| 1 | 0.4395 | 0.0366 |
| 2 | 0.1465 | 0.0049 |

## Output Files

- decoded_main_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/decoded_main_gallery.png`
- decoded_support_context_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/decoded_support_context_gallery.png`
- decoded_slot_proposal_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/decoded_slot_proposal_gallery.png`
- scalar_routing_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/scalar_routing_gallery.png`
- stage_metrics_csv: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/stage_metrics.csv`
- probe_bundle: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/val_sample0000/probe_bundle.pt`

## Stage Metrics

| stage | shape | mean deg | p95 deg | max deg | note |
| --- | --- | ---: | ---: | ---: | --- |
| lr_input | 32x32 | 2.801 | 21.391 | 59.819 | Input LR quaternion field. |
| encode_lr | 32x32 | 2.861 | 21.566 | 59.915 | Encoder output decoded back on the LR grid. |
| lr_conv1_pre_ocrp | 32x32 | 2.846 | 21.591 | 59.895 | LR spatial context just before OCRP support-bank construction. |
| selected_patch_out_hr | 128x128 | 1.636 | 1.462 | 60.231 | Hard-routed OCRP output before the HR post-conv. |
| hr_conv1_post_ocrp | 128x128 | 1.633 | 1.460 | 60.176 | HR post-conv refinement applied after OCRP assembly. |
| support_slot0_medoid_ctx | 32x32 | 3.047 | 22.048 | 59.917 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot1_medoid_ctx | 32x32 | 37.762 | 59.759 | 60.099 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot2_medoid_ctx | 32x32 | 36.204 | 53.224 | 60.257 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot0_pooled_mean | 32x32 | 2.933 | 21.908 | 59.937 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot1_pooled_mean | 32x32 | 37.748 | 59.744 | 60.395 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot2_pooled_mean | 32x32 | 36.190 | 52.987 | 60.170 | Within-slot pooled summary, averaged over HR patch tokens. |
| slot0_proposal_hr | 128x128 | 2.619 | 1.797 | 60.802 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot1_proposal_hr | 128x128 | 37.392 | 59.687 | 60.760 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot2_proposal_hr | 128x128 | 36.106 | 53.359 | 61.239 | Decoded HR patch proposal emitted by this slot before routing selection. |
| sr_output | 128x128 | 1.633 | 1.460 | 60.176 | Final decoded SR prediction. |
| hr_target | 128x128 | 0.000 | 0.000 | 0.000 | Ground-truth HR quaternion field. |
