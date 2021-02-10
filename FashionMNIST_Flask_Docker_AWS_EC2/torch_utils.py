#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:22:33 2021

@author: surekhagaikwad
"""

import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet,self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                )
        
        self.layer2 = nn.Sequential(
                    nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                )
        
        self.layer3 = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(kernel_size=3,stride=2)
                )
        
        self.fc1 = nn.Linear(128*3*3,1024)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(1024)
        self.fc2  = nn.Linear(1024,10)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.view(out.size(0),-1)
        out = self.relu(self.batch(self.fc1(out)))
        out = self.fc2(out)
        return out
  
model = FashionNet()
model.load_state_dict(torch.load("fashion-classifier.pth"))
model.eval();


def transform_image(image):
    transform = transforms.Compose([  transforms.Grayscale(num_output_channels=1),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(28,padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5],std=[0.5])
                                    ])
    
    #image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
    
