# OCRP Stage Probe

- Experiment: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01`
- Config: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/logs/run_config.json`
- Checkpoint: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/checkpoints/best_model.pt`
- Split / sample: `Val` / `0`
- Device: `cpu`
- Support grid: `32x32`
- HR shape: `128x128`
- Active slots: `[0, 1, 2]`

## Output Files

- decoded_main_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/decoded_main_gallery.png`
- decoded_support_context_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/decoded_support_context_gallery.png`
- decoded_slot_proposal_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/decoded_slot_proposal_gallery.png`
- scalar_routing_gallery: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/scalar_routing_gallery.png`
- stage_metrics_csv: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/stage_metrics.csv`
- probe_bundle: `/data/home/umang/Materials/Reynolds-QSR_4x1/experiments/IN718/iso_embedding_4x4_ocrp_01/analysis/stage_probe/val_sample0000/probe_bundle.pt`

## Stage Metrics

| stage | shape | mean deg | p95 deg | max deg | note |
| --- | --- | ---: | ---: | ---: | --- |
| lr_input | 32x32 | 2.801 | 21.391 | 59.819 | Input LR quaternion field. |
| encode_lr | 32x32 | 2.829 | 21.492 | 59.924 | Encoder output decoded back on the LR grid. |
| lr_conv1_pre_ocrp | 32x32 | 2.792 | 21.530 | 59.920 | LR spatial context just before OCRP support-bank construction. |
| selected_patch_out_hr | 128x128 | 1.647 | 1.412 | 60.124 | Hard-routed OCRP output before the HR post-conv stack. |
| hr_conv1_post_ocrp | 128x128 | 1.646 | 1.405 | 60.227 | HR conv1 refinement decoded on the HR grid. |
| hr_conv2_post_ocrp | 128x128 | 1.639 | 1.399 | 60.252 | HR conv2 refinement decoded on the HR grid. |
| hr_conv_out | 128x128 | 1.639 | 1.399 | 60.252 | Final HR post-conv stack output before decoder. |
| support_slot0_medoid_ctx | 32x32 | 2.980 | 22.117 | 59.874 | Representative medoid feature chosen for that slot on each LR support location. |
| support_slot1_medoid_ctx | 32x32 | 37.767 | 59.768 | 60.164 | Representative medoid feature chosen for that slot on each LR support location. |
| support_slot2_medoid_ctx | 32x32 | 36.202 | 53.201 | 60.166 | Representative medoid feature chosen for that slot on each LR support location. |
| support_slot0_pooled_mean | 32x32 | 2.899 | 21.881 | 59.889 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot1_pooled_mean | 32x32 | 37.752 | 59.768 | 60.005 | Within-slot pooled summary, averaged over HR patch tokens. |
| support_slot2_pooled_mean | 32x32 | 36.194 | 53.177 | 60.106 | Within-slot pooled summary, averaged over HR patch tokens. |
| slot0_proposal_hr | 128x128 | 2.586 | 1.740 | 60.617 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot1_proposal_hr | 128x128 | 44.185 | 59.716 | 60.671 | Decoded HR patch proposal emitted by this slot before routing selection. |
| slot2_proposal_hr | 128x128 | 45.660 | 58.678 | 61.403 | Decoded HR patch proposal emitted by this slot before routing selection. |
| sr_output | 128x128 | 1.639 | 1.399 | 60.252 | Final decoded SR prediction. |
| hr_target | 128x128 | 0.000 | 0.000 | 0.000 | Ground-truth HR quaternion field. |
