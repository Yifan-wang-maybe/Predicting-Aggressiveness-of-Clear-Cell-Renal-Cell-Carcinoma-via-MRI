import os
import shutil
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass, affine_transform
import matplotlib.pyplot as plt
from imageutils import *
import re



def make_dataset(dir):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    return subfolders

def volumn_normalization(orig_img, method=0, seg_img=None, const=None, excludezero=True):
    '''
    Rescale the whole image volume intensity to the range of [0, 1]
    :param orig_img:
    :param method: 0-normalization by constant, 1-percentile (0.05-99.95) normalization
    :param seg_img:
    :param const:
    :param excludezero: exclude zero voxels from normalization
    :return: normalized volume
    '''
    if method == 0:
        norm_img = orig_img.astype(np.float32) / const
        norm_img[norm_img < 0] = 0
        norm_img[norm_img > 2] = 2

    if method == 1:
        # use the whole volume to calculate percentile
        threshold = (0.05, 99.95)
        if excludezero:
            val_l, val_h = np.percentile(orig_img[orig_img > 0], threshold)  # only non-zero voxels
        else:
            val_l, val_h = np.percentile(orig_img, threshold)
        norm_img = np.copy(orig_img).astype(np.float32)
        norm_img = (norm_img - val_l) / (val_h - val_l + 1e-5)
        norm_img[norm_img < 0] = 0
        norm_img[norm_img > 1] = 1

    if method == 2:
        # TODO: Rescale the image intensity to the range of [0, 1],
        norm_img = np.copy(orig_img)

    return norm_img

def save2mhd(img, resampled_path, pid, modality, headertext):
    #def save2mhd(self, img, pid, modality):
        dstdir = os.path.join(resampled_path, pid)
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        with open(os.path.join(dstdir, f'{modality}.img'), 'wb') as f:
            img.flatten().tofile(f)
        with open(os.path.join(dstdir, f'{modality}.mhd'), 'w') as f:
            for line in headertext:
                f.write(line)
            f.write('ElementType = MET_USHORT\n')
            f.write(f'ElementDataFile = {modality}.img\n')


### 
Dataroot = 'Resamples/External/XH/HRCC'
Normalized_root_NPY = 'Ready/NPY/XH/HRCC'
Normalized_root_NTK = 'Ready/NTK/XH/HRCC'
Paths = sorted(make_dataset(Dataroot))   


for modality in ['T2W_img', 'DWI_img', 'ADC_img','T2W_seg', 'DWI_seg', 'ADC_seg']:

    for folder_path in Paths: 
        pid = folder_path[-5:]
        File_name = os.path.join(folder_path, modality)
        File_name = File_name + '.mhd'
        if os.path.exists(File_name):
            orig_img = read_mhd(File_name)
            
            print(modality)
            print(File_name)
            print(np.max(orig_img))
            print('##########################')
            
            if modality == 'DWI_img' or modality == 'T2W_img':
                norm_img = volumn_normalization(orig_img, method=1, excludezero=True)    ###DWI and T2W
            elif modality == 'ADC_img':
                norm_img = volumn_normalization(orig_img, method=0, const = 4000)         ### ADC  ### [40, 256, 256]
            elif modality == 'T2W_seg' or modality == 'DWI_seg' or modality == 'ADC_seg':
                norm_img = volumn_normalization(orig_img, method=2)         ### Copy for segmentations
            


            #####   Save as npy   #####
            dstdir = Normalized_root_NPY+'/'+pid
            
            if not os.path.exists(dstdir):
                os.makedirs(dstdir)
            
            np.save(dstdir+'/'+modality+'.npy',norm_img)    

     
            #####   Save as mhd   #####

            NewSize = 256
            NewSize_z = 40
            NewRes = 1.5
            resz = 5

            newshape = (NewSize, NewSize, NewSize_z)
                        
            headertext = 'NDims = 3\n'
            headertext += f'DimSize = {newshape[0]} {newshape[1]} {newshape[2]}\n'
            headertext += f'ElementSpacing = {NewRes:.10f} {NewRes:.10f} {resz:.10f}\n'
            #headertext += f'Position = {coord0[0]:.10f} {coord0[1]:.10f} {coord0[2]:.10f}\n'
            headertext += 'ElementByteOrderMSB = False\n'
    
            norm_img = norm_img*10000
            
            norm_img = norm_img.astype(np.int16)

            save2mhd(norm_img,Normalized_root_NTK,pid,modality,headertext)
    





  





