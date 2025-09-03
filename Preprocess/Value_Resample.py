import os
import shutil
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass, affine_transform
import matplotlib.pyplot as plt
import pandas as pd
import re
from imageutils import *



def pad_to_square(image):

    height = image.shape[1]
    width = image.shape[2]

    if height == width:
        return image  # Already square

    # Calculate padding
    diff = abs(height - width)
    pad1 = diff // 2
    pad2 = diff - pad1

    if height < width:
        padding = ((0, 0), (pad1, pad2), (0, 0)) if image.ndim == 3 else ((pad1, pad2), (0, 0))
    else:
        padding = ((0, 0),  (0, 0), (pad1, pad2)) if image.ndim == 3 else ((0, 0), (pad1, pad2))

    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_image


def make_dataset(dir):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    return subfolders


class Resample():
    NewRes, NewSize = 1.5, 256
    NewRes_z, NewSize_z = 5, 40

    def __init__(self, orig_path, resampled_path,name_to_ID,slice_registration_table):
        self.orig_path = orig_path
        self.Paths = sorted(make_dataset(self.orig_path)) 
        self.resampled_path = resampled_path
        self.slice_registration_table = slice_registration_table
        self.name_to_ID = name_to_ID


    
    def resample(self,modality):
        for folder_path in self.Paths: 
            pid = folder_path[-5:]

            pid_number = int(pid)-10000

            case_name = self.name_to_ID[pid_number]
            case_name_numbers = re.findall(r'\d+\.?\d*', case_name)
            case_name_numbers = [float(num) if '.' in num else int(num) for num in case_name_numbers] 
            case_name_numbers = case_name_numbers[0]    ### Here maybe 0 or 1,  1 for RCC1

            slice_registration = self.slice_registration_table[self.slice_registration_table['PID'] == case_name_numbers]
            if len(slice_registration)>0:
                row = slice_registration.index
                Index_compare = np.zeros(3)
                row = row[0]

                if isinstance(slice_registration.loc[row,'T2WI'],(int,float,np.int64)):
                    Index_compare[0] = slice_registration.loc[row,'T2WI']
                if isinstance(slice_registration.loc[row,'DWI'],(int,float,np.int64)):
                    Index_compare[1] = slice_registration.loc[row,'DWI']
                if isinstance(slice_registration.loc[row,'ADC'],(int,float,np.int64)):
                    Index_compare[2] = slice_registration.loc[row,'ADC']

            else:
                Index_compare = np.ones(3)
 
            File_name = os.path.join(folder_path, modality)
            File_name = File_name + '.nii'
            
            if modality[-3:] == 'seg':
                File_name = File_name + '.gz'

            if os.path.exists(File_name):


                img = sitk.ReadImage(File_name)
                img_array = sitk.GetArrayFromImage(img)  ### [Z,X,Y]
                img_array[img_array<0] = 0

                img_array = pad_to_square(img_array)

                resx, resy, resz = img.GetSpacing()


                #newshape = (self.NewSize, self.NewSize, img_array.shape[0])
                newshape = (self.NewSize, self.NewSize, self.NewSize_z)
                cz, cx, cy = img_array.shape[0] // 2, img_array.shape[1] // 2, img_array.shape[2] // 2
                x0 = cx - self.NewRes * self.NewSize / resx / 2  # get coordinate of first voxel
                y0 = cy - self.NewRes * self.NewSize / resy / 2
                if modality == 'T2W_img' or modality == 'T2W_seg':
                    z0 = Index_compare[0]
                if modality == 'DWI_img' or modality == 'DWI_seg':
                    z0 = Index_compare[1]
                if modality == 'ADC_img' or modality == 'ADC_seg':
                    z0 = Index_compare[2]

                               
                ''' Check Results'''
                #rot = np.diag([self.NewRes / resx, self.NewRes / resy, 1])  # rotation matrix is diagonal  
                rot = np.diag([self.NewRes / resx, self.NewRes / resy, self.NewRes_z/resz])
                #h_orig = get_image_header(File_name, 'all')
                #coord0 = h_orig['V2W'].dot(np.array([x0, y0, z0, 1]))  # calculate new origin world coordinate                               
                headertext = 'NDims = 3\n'
                headertext += f'DimSize = {newshape[0]} {newshape[1]} {newshape[2]}\n'
                headertext += f'ElementSpacing = {self.NewRes:.10f} {self.NewRes:.10f} {self.NewRes_z:.10f}\n'
                #headertext += f'Position = {coord0[0]:.10f} {coord0[1]:.10f} {coord0[2]:.10f}\n'
                headertext += 'ElementByteOrderMSB = False\n'
                

                
                order = 1 ###

                if modality == 'T2W_img' or modality == 'DWI_img' or modality == 'ADC_img':
                    output_type = img_array.dtype
                elif modality == 'T2W_seg' or modality == 'DWI_seg' or modality == 'ADC_seg':
                    output_type = np.float32

    
                img_resampled = affine_transform(img_array.transpose(), rot, order=order, offset=[x0, y0, z0],
                                    output_shape=newshape, output=output_type,
                                    cval=0, prefilter=False).transpose()
                  
                
                if modality == 'T2W_seg' or modality == 'DWI_seg' or modality == 'ADC_seg':
                    img_resampled[img_resampled>0] = 1
                
                
                img_resampled = img_resampled.astype(np.int16)

                
                print('############')
                print(np.max(img_resampled))
                
                '''
                #plt.imshow(img_resampled[17,:,:], cmap='gray')
                #plt.axis('off')  # Optionally turn off the axis
                #plt.show()
                '''

                self.save2mhd(img_resampled, pid, modality, headertext)
        
        return img_resampled
    
    def save2mhd(self, img, pid, modality, headertext):
    #def save2mhd(self, img, pid, modality):
        dstdir = os.path.join(self.resampled_path, pid)
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
Dataroot = 'Data/External/0424/HRCC'
resampled_root = 'Resamples/External/0424/HRCC' 

### Slice_registration_table: A Table X presents the registration of slices across T2W, DWI, and ADC modalities
### For axample, for patient 1, line 1 [3,4,5] means the third slides of T2W, the 4th slides of DWI and 5th slides of ADC should be registed together.
slice_registration_table = pd.read_excel('Slides_match/Informartion_from_point/XH.xlsx')

### A matrix map patient folder name to patient ID
name_to_ID = np.load('Data/External/0424/0424 HRCC.npy')

Process = Resample(Dataroot,resampled_root,name_to_ID,slice_registration_table)




Process.resample('T2W_seg')
print('##################################')
Process.resample('DWI_seg')
print('##################################')
Process.resample('ADC_seg')
print('##################################')

Process.resample('T2W_img')
print('##################################')
Process.resample('DWI_img')
print('##################################')
Process.resample('ADC_img')
print('##################################')


  





