import h5py
import numpy as np
import dream3d_import as d3
import glob
import os
from argparser import Argparser


args = Argparser().args

npy_file_dir = f'{args.exp_dir_path}/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    if args.data == 'Ti64_3D' or args.data == 'Ti64_3D_LR':
        int_part = filename.split("_")[7]
    elif args.data == 'Open_718':
        int_part = filename.split("_")[6]
    else: 
        int_part = filename.split("_")[8]
    return int(int_part) 

print("args.exp_type", args.exp_type)

# 
# # Updated file pattern to match the provided file path structure
# file_locs = sorted(
#     glob.glob(f'{npy_file_dir}/**/{args.data}*{args.section}*{args.exp_type}.npy', recursive=True), 
#     key=get_key
# )


# Updated file pattern to match the provided file path structure
file_locs_hr = sorted(glob.glob(f'{npy_file_dir}/**/{args.data}*{args.section}*HR**.npy', recursive=True), key=get_key)
file_locs_sr = sorted(glob.glob(f'{npy_file_dir}/**/{args.data}*{args.section}*SR**.npy', recursive=True), key=get_key)

# Combine HR and LR files
file_locs = file_locs_hr + file_locs_sr

import pdb; pdb.set_trace() 
# get all npy files location in npy_file_dir
# Get all .npy file locations in npy_file_dir (recursively)
#all_npy_files = sorted(glob.glob(f'{npy_file_dir}/**/*.npy', recursive=True))
#print("All npy files found:", all_npy_files)
#file_locs = all_npy_files
total_file = len(file_locs)
print("file locs", file_locs)

arr_list = []
for file_loc in file_locs:
    arr = np.load(file_loc)
    print(file_loc,arr.shape)
    arr_list.append(arr)

# # --- make every slice the same size (use the smallest height/width) ---
min_h = min(a.shape[0] for a in arr_list)
min_w = min(a.shape[1] for a in arr_list)

arr_list = [a[:min_h, :min_w] for a in arr_list]     # trim bottom / right

# Create a NumPy object array to hold arrays of different shapes
loaded_npy = np.array(arr_list)

#import pdb; pdb.set_trace()
loaded_npy = np.asarray(arr_list)
loaded_npy = np.float32(loaded_npy)

# Check the shape of the loaded numpy array
if args.section == 'X_normal' or args.section == 'x_normal':
    loaded_npy = np.moveaxis(loaded_npy, 0, -2)
elif args.section == 'Y_normal' or args.section == 'y_normal':
    loaded_npy = np.moveaxis(loaded_npy, 0, 1)

d3_sourceName = args.material_dream3dfile
d3source = h5py.File(d3_sourceName, 'r')
print("d3 source name", d3_sourceName)

# The path for the output Dream3D file being written.  
save_dir = f'{args.exp_dir_path}/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'
save_path = f'{save_dir}/{args.data}/Dream3D'
print("Save path:", save_path)

# Ensure the directory exists before saving
if not os.path.exists(save_path):
    os.makedirs(save_path)

d3_outputName = f'{save_path}/{args.section}_{args.file_type}.dream3d'

xdim,ydim,zdim,channeldepth = np.shape(loaded_npy)
phases = np.int32(np.ones((xdim,ydim,zdim)))
new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)


in_path = 'DataContainers/Training'
# in_path = 'DataContainers/ImageDataContainer' 
out_path = 'DataContainers/ImageDataContainer'

new_file = d3.copy_container(d3_sourceName, f'{in_path}/CellEnsembleData', d3_outputName, f'{out_path}/CellEnsembleData')

new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim),
                            source_internal_geometry_path=f'{in_path}/_SIMPL_GEOMETRY',
                            output_internal_geometry_path=f'{out_path}/_SIMPL_GEOMETRY')

new_file = d3.create_empty_container(d3_outputName, f'{out_path}/CellData', (xdim,ydim,zdim), 3)
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', loaded_npy, 'Quats')
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', phases, 'Phases')

# Close out source file to avoid weird memory errors.
d3source.close()
