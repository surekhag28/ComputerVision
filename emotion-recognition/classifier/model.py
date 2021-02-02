#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:46:49 2019

@author: surekhagaikwad
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel_BI(nn.Module):
    def __init__(self,args):
        super(CombinedModel_BI, self).__init__()
        DROP_FIRST_CLASS = 256
        NUM_CONCEPTS = 3
        NUM_CLASSES = 26

        #self.imageFeature_conv =  models.alexnet(pretrained=True).features[:12]
        resnet  = models.resnet18(pretrained=True)

        modules=list(resnet.children())[:-1]
        
        self.imageFeature_conv = nn.Sequential(*modules)
        
        for param in self.imageFeature_conv:
            param.requires_grad = True
        
        #self.image_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.bodyFeature_conv =  models.alexnet(pretrained=True).features[:12]
        #self.body_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.bodyFeature_conv = nn.Sequential(*modules)
        for param in self.bodyFeature_conv:
            param.requires_grad = True
        
        #NUM_FEATS = 12544 + 12544
        NUM_FEATS = 9216 + 9216
        
        self.fusion = nn.Sequential(
                #nn.Dropout(0.5),
                nn.Linear(1024, DROP_FIRST_CLASS),
                nn.ReLU())
        
        self.category = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        
        self.cont = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        self.sigmoid = nn.Sigmoid()
        
        self.gradients = None

     # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, imgs, imgs_body):
        
        
        out_image = self.imageFeature_conv(imgs)
        
        #out_image = self.image_max_pool(out_image)
        
        out_body = self.bodyFeature_conv(imgs_body)
        #out_body = self.body_max_pool(out_body)
        
        #out_image = out_image.view(-1,256*6*6) # 256 * 7 * 7
        #out_body = out_image.view(-1,256*6*6) # 256 * 7 * 7
        
        out_image = out_image.view(-1,512*1*1) 
        out_body = out_image.view(-1,512*1*1) 
        
        
        x = torch.cat((out_image, out_body), 1)
        x = self.fusion(x)
        out_category = self.sigmoid(self.category(x)) 
        print(out_category)
        out_cont = self.sigmoid(self.cont(x))

        return out_category, out_cont 
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.imageFeature_conv(x)


class Model_I(nn.Module):
    
    def __init__(self,args):
        super(Model_I, self).__init__()
        NUM_FEATS = 9216 #12544
        DROP_FIRST_CLASS = 256
        NUM_CONCEPTS = 3
        NUM_CLASSES = 26
        
        self.imageFeature_conv =  models.alexnet(pretrained=True).features[:12]
        
        for param in self.imageFeature_conv:
            param.requires_grad = True
        
        self.image_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.fusion = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(NUM_FEATS, 4096),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(4096,DROP_FIRST_CLASS),
                nn.ReLU())
        
        self.category = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        
        self.cont = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, imgs):
        out = self.imageFeature_conv(imgs)
        out = self.image_max_pool(out)
        
        out_image = out.view(-1,256*6*6) # 256 * 7 * 7
        
        
        out = self.fusion(out_image)
        out_category = self.sigmoid(self.category(out)) 
        out_cont = self.sigmoid(self.cont(out))
        
        return out_category,out_cont
 
class Model_B(nn.Module):

    def __init__(self,args):
        super(Model_B, self).__init__()
        NUM_FEATS = 9216 #12544
        DROP_FIRST_CLASS = 256
        NUM_CONCEPTS = 3
        NUM_CLASSES = 26

        #self.bodyFeature_conv =  models.alexnet(pretrained=True).features[:12]

        #for param in self.bodyFeature_conv:
         #   param.requires_grad = True

        self.bodyFeature_conv  = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(384),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(256))

        self.image_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        self.fusion = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(NUM_FEATS, 4096),
                nn.LeakyReLU(),
                nn.Dropout(0.25),
                nn.Linear(4096,DROP_FIRST_CLASS),
                nn.LeakyReLU())

        self.category = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)

        self.cont = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs):
        out = self.bodyFeature_conv(imgs)
        out = self.image_max_pool(out)
        out = self.avgpool(out)

        out_image = out.view(-1,256*6*6) # 256 * 7 * 7


        out = self.fusion(out_image)
        out_category = self.sigmoid(self.category(out))
        print(out_category)
        out_cont = self.sigmoid(self.cont(out))

        return out_category,out_cont
    
