1. Change user settings in build_qsr_dataset.py o build a dataset.
   1. save_dataset_ipfs saves the datasets ipf iamges.
2. Create experiments/{data_name}/{experiment_name}/config.json
   1.  This trains the model ./scripts/train.sh experiments/{data_name}/{experiment_name}
   2.  NOTE: it will overwrite the exepirment if it already exists
3.  This runs python tests 
    1.  ./scripts/run_tests.sh
