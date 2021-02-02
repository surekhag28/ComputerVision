#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:19:20 2019

@author: surekhagaikwad
"""
import numpy as np
import glob
import pandas as pd
from torch.utils.data import Dataset
from skimage import transform
import os
import torch
from PIL import Image


class CAERDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_list = glob.glob(self.root_dir+'/*.png')
        
        print(len(self.image_list))
        img_label = self.image_list[0].split('/')
        print(img_label)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
           
        img_name = self.image_list[idx]
        image = Image.open(img_name)
        img_label = img_name.split('/')
        print(img_label)
        #image.show()
    

caer = CAERDataset('./data/train')
