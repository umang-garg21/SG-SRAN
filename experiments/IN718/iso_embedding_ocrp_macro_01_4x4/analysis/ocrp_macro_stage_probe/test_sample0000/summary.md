# OCRP Macro Stage Probe

- Experiment: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01`
- Config: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/logs/run_config.json`
- Checkpoint: `experiments/IN718/iso_embedding_ocrp_macro_01/checkpoints/best_model.pt`
- Split / sample: `Test` / `0`
- Device: `cpu`
- Support grid: `32x32`
- HR shape: `128x128`
- Active slots: `[0, 1, 2]`

## Slot Usage

| slot | support coverage | HR owner usage |
| --- | ---: | ---: |
| 0 | 1.0000 | 0.9357 |
| 1 | 0.5596 | 0.0576 |
| 2 | 0.1924 | 0.0068 |

## Output Files

- decoded_main_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/decoded_main_gallery.png`
- decoded_support_context_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/decoded_support_context_gallery.png`
- decoded_slot_proposal_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/decoded_slot_proposal_gallery.png`
- scalar_routing_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/scalar_routing_gallery.png`
- stage_metrics_csv: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/stage_metrics.csv`
- probe_bundle: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_ocrp_macro_01/analysis/ocrp_macro_stage_probe/test_sample0000/probe_bundle.pt`

## Stage Metrics

| stage | shape | mean deg | p95 deg | max deg | note |
| --- | --- | ---: | ---: | ---: | --- |
| lr_input | 32x32 | 3.249 | 25.995 | 59.998 | Input LR quaternion field. |
| encode_lr | 32x32 | 3.288 | 25.897 | 59.959 | Encoder output decoded back on the LR grid. |
| lr_conv1_pre_ocrp | 32x32 | 3.271 | 25.869 | 59.995 | LR spatial context just before OCRP support-bank construction. |
| selected_patch_out_hr | 128x128 | 2.121 | 1.508 | 59.989 | Hard-routed OCRP output before the HR post-conv. |
| hr_conv1_post_ocrp | 128x128 | 2.115 | 1.518 | 59.995 | HR post-conv refinement applied after OCRP assembly. |
| support_slot0_medoid_ctx | 32x32 | 3.464 | 29.106 | 59.743 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot1_medoid_ctx | 32x32 | 41.196 | 59.614 | 59.995 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot2_medoid_ctx | 32x32 | 41.461 | 52.097 | 60.044 | Representative medoid feature chosen for that slot on each support tile. |
| support_slot0_pooled_mean | 32x32 | 3.389 | 29.160 | 59.899 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot1_pooled_mean | 32x32 | 41.200 | 59.652 | 59.995 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot2_pooled_mean | 32x32 | 41.465 | 52.144 | 59.982 | Within-slot pooled summary, averaged over HR patch tokens. |
| slot0_proposal_hr | 128x128 | 3.698 | 35.980 | 60.008 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot1_proposal_hr | 128x128 | 40.829 | 59.645 | 60.006 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot2_proposal_hr | 128x128 | 41.450 | 51.812 | 60.509 | Decoded HR patch proposal emitted by this slot before routing selection. |
| sr_output | 128x128 | 2.115 | 1.518 | 59.995 | Final decoded SR prediction. |
| hr_target | 128x128 | 0.000 | 0.000 | 0.000 | Ground-truth HR quaternion field. |
