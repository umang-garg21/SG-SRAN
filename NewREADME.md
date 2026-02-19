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


