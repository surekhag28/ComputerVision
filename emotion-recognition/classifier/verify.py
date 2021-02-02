#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:39:14 2019

@author: surekhagaikwad
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc

class calculate_precision(nn.Module):
    def __int__(self):
        super(calculate_precision, self).__init__()
        
    def forward(self,out_category,labels,args):
        
        prec = torch.zeros(1, 26,dtype=torch.float)
        
        for i in range(out_category.shape[0]):
            for j in range(out_category.shape[1]):
                if (out_category[i][j].item() >= 0.5):
                    out_category[i][j] = 1.
                else:
                    out_category[i][j] = 0.
                

        for i in range(args.n_classes):
            prec[0][i] = precision_score(torch.t(labels)[i][0:32].detach(), torch.t(out_category)[i][0:32].detach(), average='weighted')

        return prec
    
class calculate_error_rate(nn.Module):
    def __int__(self):
        super(calculate_error_rate, self).__init__()
        
    def forward(self,out_cont,labels,args):
        
        error_rate = torch.zeros(labels.shape[0], 3,dtype=torch.float)
        error_rate = torch.sum(torch.abs(out_cont - labels),0)
        
        error_rate /=labels.shape[0]
        
        return error_rate
                
    
def average_precision(out_category,labels,args):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
                    
    for i in range(args.n_classes):
        fpr[i], tpr[i], _ = roc_curve(torch.t(labels)[i][0:32], torch.t(out_category)[i][0:32])
        if(fpr[i]=='nan'):
            roc_auc[i] = auc(0, tpr[i])
        elif(tpr[i]=='nan'):
            roc_auc[i] = auc(fpr[i], 0)
        else:
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    print(roc_auc)
    

def class_wise_precision(precision, args):
    final_prec = torch.zeros(1,args.n_classes, dtype=torch.float)
    for ele in precision:
        for i in range(ele.shape[0]):
            for j in range(ele.shape[1]):
                final_prec[0][j] += ele[i][j]
    final_prec /= args.epochs
    print(final_prec)
    return final_prec  