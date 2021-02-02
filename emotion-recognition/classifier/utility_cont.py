#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:51:43 2019

@author: surekhagaikwad
"""


import torchfile
import pandas as pd
import numpy as np
import torch

train_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_train.t7')
val_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_val.t7')
test_annotations = torchfile.load('../annotations/DiscreteContinuousAnnotations26_test.t7')


train_labels = '../data/train_labels_cont.csv'
val_labels = '../data/val_labels_cont.csv'
test_labels = '../data/test_labels_cont.csv'

annotations = {'train':train_annotations, 'val':val_annotations, 'test':test_annotations}

for key in annotations.keys():
    if(key=='val' or key=='test'):
        break
    
    data = pd.DataFrame(columns=['Image','Valence','Arousal','Dominance'])
    
    for i in range(len(annotations[key])):
        filename = annotations[key][i].get(b'filename').decode('utf-8').split('.')
        file_name = filename[0]+'_'+str(i)+'.'+filename[1]
        data.loc[i] = np.array([file_name,'0','0','0'])
        print(key)
        if (key == 'test' or key == 'val'):
            annot_labels = torch.zeros(3,3, dtype=torch.int32)

            for k in range(3):
                annot_labels[k][0] = annotations[key][i].get(b'workers')[k].get(b'continuous').get(b'Valence')
                annot_labels[k][1] = annotations[key][i].get(b'workers')[k].get(b'continuous').get(b'Arousal')
                annot_labels[k][2] = annotations[key][i].get(b'workers')[k].get(b'continuous').get(b'Dominance')    
        
            labels = torch.zeros(1,3, dtype=torch.int32)

            for a in range(annot_labels.shape[0]):
                sum = 0
                for b in range(annot_labels.shape[1]):
                    sum += annot_labels[b][a].item()
                labels[0][a] = sum / annot_labels.shape[0]
                
            data.at[i,'Valence'] = labels[0][0].item()
            data.at[i,'Arousal'] = labels[0][1].item()
            data.at[i,'Dominance'] = labels[0][2].item()
        else:
            t_labels = np.zeros((1,3))
            t_labels[0][0] = annotations[key][i].get(b'workers')[0].get(b'continuous').get(b'Valence')
            t_labels[0][1] = annotations[key][i].get(b'workers')[0].get(b'continuous').get(b'Arousal')
            t_labels[0][2] = annotations[key][i].get(b'workers')[0].get(b'continuous').get(b'Dominance')
        
            data.at[i,'Valence'] = t_labels[0][0]
            data.at[i,'Arousal'] = t_labels[0][1]
            data.at[i,'Dominance'] = t_labels[0][2]
        
        print(data.iloc[i]['Valence'],data.iloc[i]['Arousal'],data.iloc[i]['Dominance'])
        
    if (key=='train'):
        data.to_csv(train_labels,sep=",", encoding="utf-8")
    elif (key=='val'):
        data.to_csv(val_labels,sep=",", encoding="utf-8")
    else:
        data.to_csv(test_labels,sep=",", encoding="utf-8")


