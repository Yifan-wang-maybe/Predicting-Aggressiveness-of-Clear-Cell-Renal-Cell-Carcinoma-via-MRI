import torch
import numpy as np
import os
import random
from PIL import Image
from imageutils import Augmentation2D, Augmentation3D_dict, Augmentation2D_dict
import matplotlib.pyplot as plt
from imageutils import *
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from pathlib import Path
from collections import defaultdict
from itertools import islice
import pickle
from scipy.ndimage import center_of_mass



def make_dataset(dir): 
    subfolders = []
    subfolders_Dataset_name = []
    subfolders_Case_name =[]
    Label = []
    for f in os.scandir(dir):      ### f.name: 
        datafoler_name = f.name
        print(datafoler_name)
        if datafoler_name in ['RCC1', 'MRIngenia', 'MR1SIGNA_premier_2', 'HS', '0424', 'XH']:
            dir_name = os.path.join(dir, f.name)       
            for f in os.scandir(dir_name):      
                P_N_Name = f.name
                P_N_dir_name = os.path.join(dir_name, P_N_Name)
                for f in os.scandir(P_N_dir_name):
                    subfolders.append(f.path)      
                    subfolders_Dataset_name.append(datafoler_name)
                    subfolders_Case_name.append(f.name)  
                    if P_N_Name == 'HRCC':
                        Label.append(1)
                    elif P_N_Name == 'LRCC':
                        Label.append(0)
                        

    return subfolders,subfolders_Dataset_name,subfolders_Case_name,Label


def Center_Crop(image,location,Seg):

    half_size = 32
    center_x = int(np.floor(float(location[0]) / 16) * 16)
    center_y = int(np.floor(float(location[1]) / 16) * 16)

    shift_x = location[0] - center_x
    shift_y = location[1] - center_y

    image_croped = image[:,:,50-shift_x-half_size:50-shift_x+half_size,50-shift_y-half_size:50-shift_y+half_size]
    Seg = Seg[:,:,50-shift_x-half_size:50-shift_x+half_size,50-shift_y-half_size:50-shift_y+half_size]

    #location_Mask = np.zeros(Seg.shape, dtype=np.int8)
    #location_Mask[:,:,center_x-half_size:center_x+half_size,center_y-half_size:center_y+half_size] = 1

    location_Mask = np.zeros([256,256], dtype=np.int8)
    location_Mask[center_x-half_size:center_x+half_size,center_y-half_size:center_y+half_size] = 1    

    return image_croped, location_Mask, Seg


class Finetune_TrainSet(torch.utils.data.Dataset):
    def __init__(self,dataroot):                
                       
        self.dataroot = dataroot
        self.subfolders,self.subfolders_Dataset_name,self.subfolders_Case_name,self.Label = make_dataset(self.dataroot)

        
        self.subfolders_positive = [a for a, l in zip(self.subfolders, self.Label) if l == 1]
        self.subfolders_Dataset_name_positive = [a for a, l in zip(self.subfolders_Dataset_name, self.Label) if l == 1]
        self.subfolders_Case_name_positive = [a for a, l in zip(self.subfolders_Case_name, self.Label) if l == 1]
        self.Label_positive = [a for a, l in zip(self.Label, self.Label) if l == 1]

        paired = list(zip(self.subfolders_positive, self.Label_positive))
        random.shuffle(paired)
        self.subfolders_positive, self.Label_positive = zip(*paired)

        self.subfolders_negative = [a for a, l in zip(self.subfolders, self.Label) if l == 0]
        self.subfolders_Dataset_name_negative = [a for a, l in zip(self.subfolders_Dataset_name, self.Label) if l == 0]
        self.subfolders_Case_name_negative = [a for a, l in zip(self.subfolders_Case_name, self.Label) if l == 0]
        self.Label_negative = [a for a, l in zip(self.Label, self.Label) if l == 0]

        paired = list(zip(self.subfolders_negative, self.Label_negative))
        random.shuffle(paired)
        self.subfolders_negative, self.Label_negative = zip(*paired)
        
        # num_positive = 36  Negative = 61

        Positive_threshold_1 = 7
        Positive_threshold_2 = 14
        Positive_threshold_3 = 21
        Positive_threshold_4 = 28

        Negative_threshold_1 = 12
        Negative_threshold_2 = 24
        Negative_threshold_3 = 36
        Negative_threshold_4 = 48


        Positive_AB_paths_train_A = self.subfolders_positive[:Positive_threshold_1]
        Positive_AB_paths_train_B = self.subfolders_positive[Positive_threshold_1:Positive_threshold_2]
        Positive_AB_paths_train_C = self.subfolders_positive[Positive_threshold_2:Positive_threshold_3]
        Positive_AB_paths_train_D = self.subfolders_positive[Positive_threshold_3:Positive_threshold_4]
        Positive_AB_paths_train_E = self.subfolders_positive[Positive_threshold_4:]

      
        Positive_Label_train_A = self.Label_positive[:Positive_threshold_1]
        Positive_Label_train_B = self.Label_positive[Positive_threshold_1:Positive_threshold_2]
        Positive_Label_train_C = self.Label_positive[Positive_threshold_2:Positive_threshold_3]
        Positive_Label_train_D = self.Label_positive[Positive_threshold_3:Positive_threshold_4]
        Positive_Label_train_E = self.Label_positive[Positive_threshold_4:]

        Negative_AB_paths_train_A = self.subfolders_negative[:Negative_threshold_1]
        Negative_AB_paths_train_B = self.subfolders_negative[Negative_threshold_1:Negative_threshold_2]
        Negative_AB_paths_train_C = self.subfolders_negative[Negative_threshold_2:Negative_threshold_3]
        Negative_AB_paths_train_D = self.subfolders_negative[Negative_threshold_3:Negative_threshold_4]
        Negative_AB_paths_train_E = self.subfolders_negative[Negative_threshold_4:]
         
        Negative_Label_train_A = self.Label_negative[:Negative_threshold_1]
        Negative_Label_train_B = self.Label_negative[Negative_threshold_1:Negative_threshold_2]
        Negative_Label_train_C = self.Label_negative[Negative_threshold_2:Negative_threshold_3]
        Negative_Label_train_D = self.Label_negative[Negative_threshold_3:Negative_threshold_4]
        Negative_Label_train_E = self.Label_negative[Negative_threshold_4:]

        
        
        
        Index_cross_validation = 1                  ################################################


        if Index_cross_validation == 0:
            np.save('Positive_AB_paths_train_A.npy',Positive_AB_paths_train_A)
            np.save('Positive_AB_paths_train_B.npy',Positive_AB_paths_train_B)
            np.save('Positive_AB_paths_train_C.npy',Positive_AB_paths_train_C)
            np.save('Positive_AB_paths_train_D.npy',Positive_AB_paths_train_D)
            np.save('Positive_AB_paths_train_E.npy',Positive_AB_paths_train_E)

            np.save('Negative_AB_paths_train_A.npy',Negative_AB_paths_train_A)
            np.save('Negative_AB_paths_train_B.npy',Negative_AB_paths_train_B)
            np.save('Negative_AB_paths_train_C.npy',Negative_AB_paths_train_C)
            np.save('Negative_AB_paths_train_D.npy',Negative_AB_paths_train_D)
            np.save('Negative_AB_paths_train_E.npy',Negative_AB_paths_train_E)

        if Index_cross_validation == 0:
        
            self.AB_paths_train = Positive_AB_paths_train_A + Positive_AB_paths_train_B + Positive_AB_paths_train_C + Positive_AB_paths_train_D + Negative_AB_paths_train_A + Negative_AB_paths_train_B + Negative_AB_paths_train_C + Negative_AB_paths_train_D
            self.Label_train = Positive_Label_train_A + Positive_Label_train_B + Positive_Label_train_C + Positive_Label_train_D + Negative_Label_train_A + Negative_Label_train_B + Negative_Label_train_C + Negative_Label_train_D

            self.AB_paths_val = Positive_AB_paths_train_E + Negative_AB_paths_train_E
            self.Label_val = Positive_Label_train_E + Negative_Label_train_E

        elif Index_cross_validation == 1:

            self.AB_paths_train = Positive_AB_paths_train_A + Positive_AB_paths_train_B + Positive_AB_paths_train_C + Positive_AB_paths_train_E + Negative_AB_paths_train_A + Negative_AB_paths_train_B + Negative_AB_paths_train_C + Negative_AB_paths_train_E
            self.Label_train = Positive_Label_train_A + Positive_Label_train_B + Positive_Label_train_C + Positive_Label_train_E + Negative_Label_train_A + Negative_Label_train_B + Negative_Label_train_C + Negative_Label_train_E

            self.AB_paths_val = Positive_AB_paths_train_D + Negative_AB_paths_train_D
            self.Label_val = Positive_Label_train_D + Negative_Label_train_D

        elif Index_cross_validation == 2:

            self.AB_paths_train = Positive_AB_paths_train_A + Positive_AB_paths_train_B + Positive_AB_paths_train_D + Positive_AB_paths_train_E + Negative_AB_paths_train_A + Negative_AB_paths_train_B + Negative_AB_paths_train_D + Negative_AB_paths_train_E
            self.Label_train = Positive_Label_train_A + Positive_Label_train_B + Positive_Label_train_D + Positive_Label_train_E + Negative_Label_train_A + Negative_Label_train_B + Negative_Label_train_D + Negative_Label_train_E

            self.AB_paths_val = Positive_AB_paths_train_C + Negative_AB_paths_train_C
            self.Label_val = Positive_Label_train_C + Negative_Label_train_C

        elif Index_cross_validation == 3:

            self.AB_paths_train = Positive_AB_paths_train_A + Positive_AB_paths_train_C + Positive_AB_paths_train_D + Positive_AB_paths_train_E + Negative_AB_paths_train_A + Negative_AB_paths_train_C + Negative_AB_paths_train_D + Negative_AB_paths_train_E
            self.Label_train = Positive_Label_train_A + Positive_Label_train_C + Positive_Label_train_D + Positive_Label_train_E + Negative_Label_train_A + Negative_Label_train_C + Negative_Label_train_D + Negative_Label_train_E

            self.AB_paths_val = Positive_AB_paths_train_B + Negative_AB_paths_train_B
            self.Label_val = Positive_Label_train_B + Negative_Label_train_B


        elif Index_cross_validation == 4:

            self.AB_paths_train = Positive_AB_paths_train_B + Positive_AB_paths_train_C + Positive_AB_paths_train_D + Positive_AB_paths_train_E + Negative_AB_paths_train_B + Negative_AB_paths_train_C + Negative_AB_paths_train_D + Negative_AB_paths_train_E
            self.Label_train = Positive_Label_train_B + Positive_Label_train_C + Positive_Label_train_D + Positive_Label_train_E + Negative_Label_train_B + Negative_Label_train_C + Negative_Label_train_D + Negative_Label_train_E

            self.AB_paths_val = Positive_AB_paths_train_A + Negative_AB_paths_train_A
            self.Label_val = Positive_Label_train_A + Negative_Label_train_A    


        self.Whole_Data = []
        self.Whole_mask = []
        self.Whole_location = []
        self.Whole_label = []

        half_size = 50
        
        for i in range(len(self.AB_paths_train)):
            name = self.AB_paths_train[i]
            label = self.Label_train[i]
        
            T2W_img = np.load(os.path.join(name,'T2W_img.npy'))
            ADC_img = np.load(os.path.join(name,'ADC_img.npy'))
            DWI_img = np.load(os.path.join(name,'DWI_img.npy'))
            
            T2W_seg = np.load(os.path.join(name,'T2W_seg.npy'))
            ADC_seg = np.load(os.path.join(name,'ADC_seg.npy'))
            DWI_seg = np.load(os.path.join(name,'DWI_seg.npy'))   

            
            struct3 = ndimage.generate_binary_structure(3, 3)
            mask_label, n_orig_A = ndimage.label(T2W_seg, struct3)
           
            if n_orig_A > 1:
                sizes = ndimage.sum(np.ones_like(T2W_seg), mask_label, index=range(1, n_orig_A + 1))
                index = np.argmax(sizes)
                mask_label[mask_label == index+1] = 10
                mask_label[mask_label < 10] = 0
                mask_label[mask_label == 10] = 1


            center = np.array(center_of_mass(mask_label))

            x = int(center[1])
            y = int(center[2])

        
            labeled_indices = np.where(mask_label == 1)
            z_min = np.min(labeled_indices[0])
            z_max = np.max(labeled_indices[0])
            #z_mid = (z_max-z_min)//2

            z_mid = int(center[0])

            pad_width = [(0, 0), (half_size, half_size), (half_size, half_size)]
            T2W_img_pad = np.pad(T2W_img, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_img_pad = np.pad(ADC_img, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_img_pad = np.pad(DWI_img, pad_width=pad_width, mode='constant', constant_values=0)

            T2W_seg_pad = np.pad(T2W_seg, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_seg_pad = np.pad(ADC_seg, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_seg_pad = np.pad(DWI_seg, pad_width=pad_width, mode='constant', constant_values=0)

            #T2W_img = T2W_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            #T2W_img = T2W_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]

            T2W_img = T2W_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            ADC_img = ADC_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            DWI_img = DWI_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]

            image = np.concatenate((T2W_img[np.newaxis,:,:,:],ADC_img[np.newaxis,:,:,:],DWI_img[np.newaxis,:,:,:]),axis=0)

            mask_slices_label = T2W_seg_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            image_crop, location_mask_crop, mask_slices_label_crop = Center_Crop(image,[x,y],mask_slices_label[np.newaxis,:,:,:]) 

            
            self.Whole_Data.append(image_crop)
            self.Whole_mask.append(mask_slices_label_crop)
            self.Whole_location.append(location_mask_crop)
            self.Whole_label.append(label)


        print('Training Set Size',len(self.Whole_Data))
               
        self.dataaugment = Augmentation2D_dict(**{'flipX':0.5, 'rotxy':30, 'translation': [10, 10], 'spatialscale': [0.1, 0.1], 'intensityscale': 0.05,'noise': 0.05,
                                                'gamma': [0.9, 1.11], 'spatial':['image','seg'], 'intensity': ['image']}) 
                

    def __len__(self):
        return len(self.Whole_Data)

    def Get_val_index(self):
        return self.AB_paths_val, self.Label_val


    def __getitem__(self, idx):

                
        lesion_images = self.Whole_Data[idx]                # [3,Slices,64,64]
        lesion_mask_seg = self.Whole_mask[idx]  
        
        image = lesion_images[:,0,:,:]
        Mask_seg = lesion_mask_seg[:,0,:,:]

        number_of_slides = lesion_images.shape[1]

        for i in range(number_of_slides-1):
            k = i+1
            image = np.concatenate((image, lesion_images[:,k,:,:]), axis=0)
            Mask_seg = np.concatenate((Mask_seg, lesion_mask_seg[:,k,:,:]), axis=0)


        image = image.astype(np.float32)
        Mask_seg = Mask_seg.astype(np.float32)
        x = {'image': image,'seg':Mask_seg}
        x_trans = self.dataaugment(x)
        image = x_trans['image']
        seg = x_trans['seg']
        image = image.astype(np.float16)
        seg = seg.astype(np.float16)


        lesion_label = np.float16(self.Whole_label[idx])
        Mask_location = self.Whole_location[idx]
       
        return image,seg,lesion_label, Mask_location
    



class Finetune_ValSet(torch.utils.data.Dataset):
    def __init__(self,dataroot,AB_paths_val,label_val):                
                       
        '''
        self.dataroot = dataroot
        self.subfolders,self.subfolders_Dataset_name,self.subfolders_Case_name,self.Label = make_dataset(self.dataroot)

        self.AB_paths_train = self.subfolders[:7] + self.subfolders[10:26] 
        self.Label_train = self.Label[:7] + self.Label[10:26] 

        self.AB_paths_val = self.subfolders[7:10] + self.subfolders[26:] 
        self.Label_val = self.Label[7:10] + self.Label[26:] 
        '''  

        self.AB_paths_val = AB_paths_val
        self.Label_val = label_val


        self.Whole_Data = []
        self.Whole_mask = []
        self.Whole_location = []
        self.Whole_label = []

        half_size = 50
        
        for i in range(len(self.AB_paths_val)):
            name = self.AB_paths_val[i]
            label = self.Label_val[i]
        
            T2W_img = np.load(os.path.join(name,'T2W_img.npy'))
            ADC_img = np.load(os.path.join(name,'ADC_img.npy'))
            DWI_img = np.load(os.path.join(name,'DWI_img.npy'))
            
            T2W_seg = np.load(os.path.join(name,'T2W_seg.npy'))
            ADC_seg = np.load(os.path.join(name,'ADC_seg.npy'))
            DWI_seg = np.load(os.path.join(name,'DWI_seg.npy'))   

            
            struct3 = ndimage.generate_binary_structure(3, 3)
            mask_label, n_orig_A = ndimage.label(T2W_seg, struct3)
           
            if n_orig_A > 1:
                sizes = ndimage.sum(np.ones_like(T2W_seg), mask_label, index=range(1, n_orig_A + 1))
                index = np.argmax(sizes)
                mask_label[mask_label == index+1] = 10
                mask_label[mask_label < 10] = 0
                mask_label[mask_label == 10] = 1


            center = np.array(center_of_mass(mask_label))

            x = int(center[1])
            y = int(center[2])

        
            labeled_indices = np.where(mask_label == 1)
            z_min = np.min(labeled_indices[0])
            z_max = np.max(labeled_indices[0])
            #z_mid = (z_max-z_min)//2

            z_mid = int(center[0])

            pad_width = [(0, 0), (half_size, half_size), (half_size, half_size)]
            T2W_img_pad = np.pad(T2W_img, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_img_pad = np.pad(ADC_img, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_img_pad = np.pad(DWI_img, pad_width=pad_width, mode='constant', constant_values=0)

            T2W_seg_pad = np.pad(T2W_seg, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_seg_pad = np.pad(ADC_seg, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_seg_pad = np.pad(DWI_seg, pad_width=pad_width, mode='constant', constant_values=0)

            #T2W_img = T2W_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            #T2W_img = T2W_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]

            T2W_img = T2W_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            ADC_img = ADC_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            DWI_img = DWI_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]

            image = np.concatenate((T2W_img[np.newaxis,:,:,:],ADC_img[np.newaxis,:,:,:],DWI_img[np.newaxis,:,:,:]),axis=0)

            mask_slices_label = T2W_seg_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            image_crop, location_mask_crop, mask_slices_label_crop = Center_Crop(image,[x,y],mask_slices_label[np.newaxis,:,:,:]) 

            
            self.Whole_Data.append(image_crop)
            self.Whole_mask.append(mask_slices_label_crop)
            self.Whole_location.append(location_mask_crop)
            self.Whole_label.append(label)


        print('Val Set Size',len(self.Whole_Data))
               
        #self.dataaugment = Augmentation2D_dict(**{'flipX':0.5, 'rotxy':30, 'translation': [10, 10], 'spatialscale': [0.1, 0.1], 'intensityscale': 0.05,'noise': 0.05,
        #                                        'gamma': [0.9, 1.11], 'spatial':['image','seg'], 'intensity': ['image']}) 
                

    def __len__(self):
        return len(self.Whole_Data)

    def Get_val_index(self):
        return self.AB_paths_val, self.Label_val


    def __getitem__(self, idx):

                
        lesion_images = self.Whole_Data[idx]                # [3,Slices,64,64]
        lesion_mask_seg = self.Whole_mask[idx]  
        
        image = lesion_images[:,0,:,:]
        Mask_seg = lesion_mask_seg[:,0,:,:]

        number_of_slides = lesion_images.shape[1]

        for i in range(number_of_slides-1):
            k = i+1
            image = np.concatenate((image, lesion_images[:,k,:,:]), axis=0)
            Mask_seg = np.concatenate((Mask_seg, lesion_mask_seg[:,k,:,:]), axis=0)


        image = image.astype(np.float32)
        seg = Mask_seg.astype(np.float32)
        
        '''
        x = {'image': image,'seg':Mask_seg}
        x_trans = self.dataaugment(x)
        image = x_trans['image']
        seg = x_trans['seg']
        image = image.astype(np.float16)
        seg = seg.astype(np.float16)
        '''

        lesion_label = np.float16(self.Whole_label[idx])
        Mask_location = self.Whole_location[idx]
       
        return image,seg,lesion_label, Mask_location



class Finetune_TestSet(torch.utils.data.Dataset):
    def __init__(self,dataroot):                
                       
        
        self.dataroot = dataroot
        self.subfolders,self.subfolders_Dataset_name,self.subfolders_Case_name,self.Label = make_dataset(self.dataroot)
        

        self.AB_paths_val = self.subfolders
        self.Label_val = self.Label

        print(self.AB_paths_val)


        self.Whole_Data = []
        self.Whole_mask = []
        self.Whole_location = []
        self.Whole_label = []

        half_size = 50
        
        for i in range(len(self.AB_paths_val)):
            name = self.AB_paths_val[i]
            label = self.Label_val[i]
        
            T2W_img = np.load(os.path.join(name,'T2W_img.npy'))
            ADC_img = np.load(os.path.join(name,'ADC_img.npy'))
            DWI_img = np.load(os.path.join(name,'DWI_img.npy'))
            
            T2W_seg = np.load(os.path.join(name,'T2W_seg.npy'))
            ADC_seg = np.load(os.path.join(name,'ADC_seg.npy'))
            DWI_seg = np.load(os.path.join(name,'DWI_seg.npy'))   

            
            struct3 = ndimage.generate_binary_structure(3, 3)
            mask_label, n_orig_A = ndimage.label(T2W_seg, struct3)
           
            if n_orig_A > 1:
                sizes = ndimage.sum(np.ones_like(T2W_seg), mask_label, index=range(1, n_orig_A + 1))
                index = np.argmax(sizes)
                mask_label[mask_label == index+1] = 10
                mask_label[mask_label < 10] = 0
                mask_label[mask_label == 10] = 1


            center = np.array(center_of_mass(mask_label))

            x = int(center[1])
            y = int(center[2])

        
            labeled_indices = np.where(mask_label == 1)
            z_min = np.min(labeled_indices[0])
            z_max = np.max(labeled_indices[0])
            #z_mid = (z_max-z_min)//2

            z_mid = int(center[0])

            pad_width = [(0, 0), (half_size, half_size), (half_size, half_size)]
            T2W_img_pad = np.pad(T2W_img, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_img_pad = np.pad(ADC_img, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_img_pad = np.pad(DWI_img, pad_width=pad_width, mode='constant', constant_values=0)

            T2W_seg_pad = np.pad(T2W_seg, pad_width=pad_width, mode='constant', constant_values=0)
            ADC_seg_pad = np.pad(ADC_seg, pad_width=pad_width, mode='constant', constant_values=0)
            DWI_seg_pad = np.pad(DWI_seg, pad_width=pad_width, mode='constant', constant_values=0)

            #T2W_img = T2W_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            #T2W_img = T2W_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #ADC_img = ADC_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]
            #DWI_img = DWI_img_pad[z_mid-1:z_mid+2, x:x+half_size*2,y:y+half_size*2]

            T2W_img = T2W_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            ADC_img = ADC_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]
            DWI_img = DWI_img_pad[z_mid:z_mid+1, x:x+half_size*2,y:y+half_size*2]

            image = np.concatenate((T2W_img[np.newaxis,:,:,:],ADC_img[np.newaxis,:,:,:],DWI_img[np.newaxis,:,:,:]),axis=0)

            mask_slices_label = T2W_seg_pad[np.maximum(z_min-1,0):np.minimum(z_max+1,49),x:x+half_size*2,y:y+half_size*2]

            image_crop, location_mask_crop, mask_slices_label_crop = Center_Crop(image,[x,y],mask_slices_label[np.newaxis,:,:,:]) 

            
            self.Whole_Data.append(image_crop)
            self.Whole_mask.append(mask_slices_label_crop)
            self.Whole_location.append(location_mask_crop)
            self.Whole_label.append(label)


        print('Testing Set Size',len(self.Whole_Data))
               
        #self.dataaugment = Augmentation2D_dict(**{'flipX':0.5, 'rotxy':30, 'translation': [10, 10], 'spatialscale': [0.1, 0.1], 'intensityscale': 0.05,'noise': 0.05,
        #                                        'gamma': [0.9, 1.11], 'spatial':['image','seg'], 'intensity': ['image']}) 
                

    def __len__(self):
        return len(self.Whole_Data)

    def Get_val_index(self):
        return self.AB_paths_val, self.Label_val


    def __getitem__(self, idx):

                
        lesion_images = self.Whole_Data[idx]                # [3,Slices,64,64]
        lesion_mask_seg = self.Whole_mask[idx]  
        
        image = lesion_images[:,0,:,:]
        Mask_seg = lesion_mask_seg[:,0,:,:]

        number_of_slides = lesion_images.shape[1]

        for i in range(number_of_slides-1):
            k = i+1
            image = np.concatenate((image, lesion_images[:,k,:,:]), axis=0)
            Mask_seg = np.concatenate((Mask_seg, lesion_mask_seg[:,k,:,:]), axis=0)


        image = image.astype(np.float32)
        seg = Mask_seg.astype(np.float32)
        
        '''
        x = {'image': image,'seg':Mask_seg}
        x_trans = self.dataaugment(x)
        image = x_trans['image']
        seg = x_trans['seg']
        image = image.astype(np.float16)
        seg = seg.astype(np.float16)
        '''

        lesion_label = np.float16(self.Whole_label[idx])
        Mask_location = self.Whole_location[idx]
       
        return image,seg,lesion_label, Mask_location
    




