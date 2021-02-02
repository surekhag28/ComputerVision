#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:03:15 2019

@author: surekhagaikwad
"""

import torchfile
import pandas as pd
import numpy as np
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_train.t7')
val_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_val.t7')
test_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_test.t7')
#print(val_annotations)

train_labels = '../data/train_labels.csv'
val_labels = '../data/val_labels.csv'
test_labels = '../data/test_labels.csv'

annotations = {'train':train_annotations, 'val':val_annotations, 'test':test_annotations}

for key in annotations.keys():
    print(key)
    if (key=='test' or key=='val'):
        continue
    
    data = pd.DataFrame(columns=['Image','Peace','Affection','Esteem','Anticipation','Engagement','Confidence','Happiness','Pleasure',
          'Excitement','Surprise','Sympathy','Doubt/Confusion','Disconnection','Fatigue','Embarrassment','Yearning',
          'Disapproval','Aversion','Annoyance','Anger','Sensitivity','Sadness','Disquietment','Fear',
          'Pain','Suffering'])
    
    if (key=='train'):
        path = '../data/train/'
    elif (key=='val'):  
        path = '../data/val/'
    else:
        path = '../data/test/'
            
    for i in range(len(annotations[key])):
        img_path = '../datasets/'
        img_path = img_path + annotations[key][i].get(b'folder').decode('utf-8')+'/'+annotations[key][i].get(b'filename').decode('utf-8')
        img = Image.open(img_path)
        
            
        if (img.mode=='RGBA'):
            img = img.convert('RGB')
            
        file_path = annotations[key][i].get(b'filename').decode('utf-8').split('.')
        file_name = file_path[0]+'_'+str(i)+'.'+file_path[1]
        img.save(path+file_path[0]+'_'+str(i)+'.'+file_path[1])
        data.loc[i] = np.array([file_name,'0','0','0','0','0','0','0','0','0','0','0','0','0',
                                 '0','0','0','0','0','0','0','0','0','0','0','0','0'])
        
        labels = torch.zeros(1,26, dtype=torch.int32)
        if(key=='test' or key=='val'):
           
           label_list = []
           for k in range(3):
               values = annotations[key][i].get(b'workers')[k].get(b'labels')
           for j in range(len(values)):
               label_list.append(values[j])
           #print(label_list)
           max_value = max(label_list)      
           
           for j in range(len(label_list)):
               index = label_list[j]
               div_value = float(label_list[j])/float(max_value)
               if(div_value>=0.5):
                   labels[0][index-1] = 1
           for j in range(len(labels[0])):
               data.iat[i,j+1] = labels[0][j].item()
        else:
            labels = annotations[key][i].get(b'workers')[0].get(b'labels')
        
            for j in range(len(labels)):
                k = labels[j]
                data.iat[i,k] = 1
            
        print(img_path,':- ',labels)
    if (key=='train'):
        data.to_csv(train_labels,sep=",", encoding="utf-8")
    elif (key=='val'):
        data.to_csv(val_labels,sep=",", encoding="utf-8")
    else:
        data.to_csv(test_labels,sep=",", encoding="utf-8")
        
    print(len(annotations[key]))
        
    
