import h5py
import numpy as np
import glob
import os
from PIL import Image 
from argparser import Argparser

args = Argparser().args
exp_dir_path= args.exp_dir_path
npy_file_dir = f'{exp_dir_path}/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

#file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}/*{args.section}*{args.file_type}*.npy'))

def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    if args.data == 'Ti64_3D' or args.data == 'Ti64_3D_LR':
        int_part = filename.split("_")[7]
    elif args.data == 'Open_718':
        int_part = filename.split("_")[6]
    else: 
        int_part = filename.split("_")[8]

    return int(int_part) 


#file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}*{args.exp_type}.npy'), key=get_key)
file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}*HR**.npy'), key=get_key)
file_locs += sorted(glob.glob(f'{npy_file_dir}/{args.data}*SR**.npy'), key=get_key)

#file_locs = glob.glob(f'{npy_file_dir}/**/*.npy', recursive=True)

total_file = len(file_locs)
print("total files", total_file)    

dream_3d_file = f'{npy_file_dir}/{args.data}/Dream3D/{args.section}_{args.file_type}.dream3d'
print("dream3Dfile", dream_3d_file)

# check if there is a file at dream3Dfile location
if not os.path.exists(dream_3d_file):
    print(f"File {dream_3d_file} does not exist")
    exit()

dream3d_file = h5py.File(f'{dream_3d_file}', 'r')
print("LOADED THE FILE")

img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']
#img = dream3d_file['DataContainers']['Test']['CellData']['IPFColor']

if args.section == 'X_normal' or args.section == 'x_normal':
    img = np.moveaxis(img, -2, 0)    
elif args.section == 'Y_normal' or args.section == 'y_normal':
    img = np.moveaxis(img, 1, 0)

for i, file_loc in enumerate(file_locs):
   
    basename = os.path.basename(file_loc)
    filename = os.path.splitext(basename)[0]
    #import pdb; pdb.set_trace()
    image = Image.fromarray(img[i,:,:,:], "RGB")
    print("the path to save is", f'{npy_file_dir}/{args.data}/Dream3D/{filename}.png')
    image.save(f'{npy_file_dir}/{args.data}/Dream3D/{filename}.png')