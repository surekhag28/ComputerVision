#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:39:08 2019

@author: surekhagaikwad
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model

def show_featuremap(train_loader,args,dev):
    dataiter = iter(train_loader)
    images, images_body, labels, labels_body = dataiter.next()
    imshow_image(images[0],labels[0])
    imshow_image(images_body[0],labels_body[0])
    
    img = images[0:1]
    print(img.shape)
    img_body = images_body[0:1]
    
    model = load_checkpoint('emotion_model.pth') 
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    for name, m in model.named_modules():
        if(name=='imageFeature_conv.10'):
            m.register_forward_hook(get_activation(name))
            
            if(args.model_type=='I'):
                output = model(img.to(dev))
            elif(args.model_type=='B'):
                output = model(img.to(dev))
            elif(args.model_type=='BI'):
                output = model(img.to(dev), img_body.to(dev))
                
            act = activation[name]
            
            for idx in range(4):
                filename = './feature_'+str(idx)+'.jpg'
                plt.figure(figsize=(6,5))
                img = act[0][idx]
                print(act[0].shape)
                img = torchvision.transforms.ToPILImage()(img.to('cpu'))
                img = torchvision.transforms.Grayscale(3)(img)
                img = torchvision.transforms.Resize([227,227])(img)
                img.save(filename)
                plt.imshow(img)
            break
    
def imshow_image(image,label):

  fig=plt.figure()
  fig.add_subplot(1, 1, 1)
  #plt.title(label)
  mean = torch.tensor([0.5,0.5,0.5])
  std = torch.tensor([0.5,0.5,0.5])
  for img, m, s in zip(image, mean, std):
      img.mul_(s).add_(m)
  
  img = torchvision.transforms.ToPILImage()(img)
  plt.imshow(img)
  plt.show()
    