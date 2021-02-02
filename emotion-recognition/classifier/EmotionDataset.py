#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:47:03 2019

@author: surekhagaikwad
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import transform
import os
from PIL import Image, ImageFile
import torchfile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EmotionDataset(Dataset):
    
    def __init__(self, disc_file, cont_file, root_dir, flag, transform=transform):
        
        self.disc_labels = pd.read_csv(disc_file)
        self.cont_labels = pd.read_csv(cont_file)
        self.root_dir = root_dir
        self.flag = flag
        self.transform = transform
        
        self.train_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_train.t7')
        self.val_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_val.t7')
        self.test_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_test.t7')
        
    def __len__(self):
        return len(self.disc_labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        filename = self.disc_labels.iloc[idx,1].split('.')
        fname = filename[0][0:filename[0].rfind('_')]+'.'+filename[1]
        
        img_path = os.path.join(self.root_dir,self.disc_labels.iloc[idx,1])
        image = Image.open(img_path)
        #image.show()
        
        if(self.flag == 'train'):
            annotations = self.train_annotations
        elif(self.flag == 'test'):
            annotations = self.test_annotations
        elif(self.flag == 'val'):
            annotations = self.val_annotations
            
        for i in range(len(annotations)):
            
            if(annotations[i].filename.decode('utf-8') == fname):
                width,height = image.size

                body_x1=max(1,annotations[i].body_bbox[0])-1
                body_y1=max(1,annotations[i].body_bbox[1])-1
                body_x2=min(annotations[i].body_bbox[2],width)
                body_y2=min(annotations[i].body_bbox[3],height)
            
                image_body = image.crop((body_x1,body_y1,body_x2,body_y2))
                #image_body.show()
                break
        
        labels = self.disc_labels.iloc[idx, 2:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 26)
        labels = torch.from_numpy(labels)
        labels = torch.squeeze(labels,0)
        
        labels_cont = self.cont_labels.iloc[idx, 2:]
        labels_cont = np.array([labels_cont])
        labels_cont = labels_cont.astype('float').reshape(-1, 3) 
        labels_cont = torch.from_numpy(labels_cont)
        labels_cont = torch.squeeze(labels_cont,0)
        
        if self.transform:
            image = self.transform(image)
            image_body = self.transform(image_body)
        
        return (image,image_body,labels,labels_cont)

        
        
        
    