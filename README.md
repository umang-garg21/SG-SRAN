1. Change user settings in build_qsr_dataset.py o build a dataset.
   1. save_dataset_ipfs saves the datasets ipf iamges.
2. Create experiments/{data_name}/{experiment_name}/config.json
   1.  This trains the model ./scripts/train.sh experiments/{data_name}/{experiment_name}
   2. (optional) To run multi-gpu DDP framework
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/train_ddp.sh experiments/{data_name}/{experiment_name} {num_gpus}
   2.  NOTE: it will overwrite the exepirment if it already exists
3.  This runs python tests 
   1.  ./scripts/run_tests.sh
4.  
scripts/train.sh /home/warren/projects/Reynolds-QSR/experiments/IN718/debug_x4 --config config_smoke.json

scripts/train_autoencoder.sh /home/warren/projects/Reynolds-QSR/experiments/IN718/debug_x4 --config config_smoke.json

scripts/train_autoencoder.sh /home/warren/projects/Reynolds-QSR/experiments/IN718/LAE_01 --config config_tiny.json

scripts/train.sh /home/warren/projects/Reynolds-QSR/experiments/IN718/invariant_sr


cd /home/warren/projects/Reynolds-QSR && source ~/anaconda3/etc/profile.d/conda.sh && conda activate sym-QSR && python scripts/debug_autoencoder_viz.py --exp_dir experiments/IN718/LAE_01 --config config_tiny_physics.json --checkpoint last --split test --sample_idx 0



python scripts/export_decoder_teacher_table.py --exp_dir experiments/IN718/debug_x4 --config config.json --split Train --out experiments/IN718/debug_x4/logs/teacher_table_train_highacc.pt --decoder_backend lookup --decoder_lookup_resolution 1 --decoder_lookup_npy_path fast_lookup_res1.npy --decoder_lookup_refine_steps 40 --decoder_lookup_refine_lr 0.001 --decoder_w6 0.5 --include_cubochoric_fz --cubochoric_resolution 1 --cubochoric_method cubochoric


python scripts/train_decoder_distill.py \
  --teacher_table experiments/IN718/debug_x4/logs/teacher_table_train_highacc.pt \
  --out experiments/IN718/debug_x4/checkpoints/decoder_distill_best.pt \
  --epochs 120 \
  --batch_size 16384 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --hidden_dim 768 \
  --num_layers 6 \
  --dropout 0.0 \
  --val_ratio 0.1