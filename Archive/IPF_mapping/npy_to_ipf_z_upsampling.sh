#!/bin/bash
# INSTRUCTIONS:
# 1. Open the project folder as the main directory
# 2. modify the variables below as needed
# Variables:
# modelname: name of the model
# exp_type: type of experiment
# modeltoload: model to load
# filetype: file type: SR or HR
# datasettype: dataset type: Test or Val
# materials: specify the material for right symmetry
# sect: which axis for sectioning and superresolution
# exp_dir_path: path to the experiment directory, all the relevant model files 
# 		-------and corresponding quaternion (./Quaternion_experiments/results ...)
#		-------files should be in this directory
# material_dream3dfile: path to the correspoding material dream3d file
# 3. Run the script

current_dir=$(pwd)
modelname="test_model"
exp_type="minimum_angle_orientation"
modeltoload="model_best"
filetype="HR"
datasettype=("Test")
materials=("Open_718")
sect="x_normal"
exp_dir_path="$current_dir/Quaternion_experiments"
material_dream3dfile="/data/home/warrenz/Materials/Materials_data_mount/Open_718_Training.dream3d"


for material in ${materials[@]};do
	for d_type in ${datasettype[@]}; do
		for s in ${sect[@]}; do
			echo "$d_type  $s" 

			echo "Running Numpy to Dream3D"
			python ./IPF_mapping/npy_to_dream3d.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s --exp_type $exp_type --exp_dir_path $exp_dir_path --material_dream3dfile $material_dream3dfile
			
			echo "Changing Variable in JSON "
			python ./IPF_mapping/change_var_in_json.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s --exp_type $exp_type --exp_dir_path $exp_dir_path --material_dream3dfile $material_dream3dfile 

			echo "Running Dream 3D Pipeline"		

			#path to Dream3D program
			./IPF_mapping/DREAM3D-6.5.171-Linux-x86_64/bin/PipelineRunner -p $current_dir/IPF_mapping/pipeline.json

			echo "Running Dream 3D Pipeline"
            python ./IPF_mapping/dream3d_to_rgb.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s --exp_type $exp_type --exp_dir_path $exp_dir_path --material_dream3dfile $material_dream3dfile
		
		done
	done

done
