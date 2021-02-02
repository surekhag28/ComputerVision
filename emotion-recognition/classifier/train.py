#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:42:06 2019

@author: surekhagaikwad
"""

import torch
import torch.nn as nn
from verify import calculate_precision,class_wise_precision,average_precision,calculate_error_rate
from visualize import show_featuremap

dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Loss_for_discrete(nn.Module):
  def __init__(self):
    super().__init__()
    self.nF = torch.tensor(1.2)
    self.N = 26
    self.nW = 1
    self.histClasses = torch.zeros(1,self.N)
    self.class_weights = torch.ones(self.N)
    
  def forward(self,out_category,labels):
      self.histClasses = torch.sum(torch.gt(labels,0),0).unsqueeze(dim=0)
      self.normHist = self.histClasses
      
      for i in range(self.N):
          if self.histClasses[0][i].item() < 1:
              self.class_weights[i] = 0.0001
          else:
              if(self.nW > 0):
                  self.class_weights[i] = 1 / (torch.log(self.nF + self.normHist[0][i].item()))
              else:
                  self.class_weights[i] = 1 / (self.normHist[0][i].item())
                  
      self.class_weights = self.class_weights.to(dev)
      #loss = torch.sqrt((self.class_weights * torch.pow((out_category - labels),2)).sum()) / self.N
      
      loss = (self.class_weights * torch.pow((out_category - labels),2)).sum() / self.N
      
      return loss


class Loss_for_continuous(nn.Module):
  def __init__(self):
    super().__init__()
    self.C = 3
    self.theta = 0.1
    
  def forward(self,out_cont,labels_body):
      v = torch.empty(out_cont.shape[0], out_cont.shape[1], dtype=torch.float)
      
      diff = torch.abs(out_cont - labels_body)
      
      for i in range(diff.shape[0]):
          for j in range(diff.shape[1]):
              if (diff[i][j] < self.theta):
                  v[i][j] = 0.1
              else:
                  v[i][j] = 1.
                  
      v = v.to(dev)
      #loss = torch.sqrt((v * torch.pow((out_cont - labels_body),2)).sum()) / self.C
      
      loss = (v * torch.pow((out_cont - labels_body),2)).sum() / self.C
      return loss


criterion_disc = Loss_for_discrete()
criterion_cont = Loss_for_continuous()
cal_precision = calculate_precision()
cal_error_rate = calculate_error_rate()


train_losses, val_losses = [], []
train_precision, val_precision = [],[]
 

def train(args, model, optimizer, dataloaders):
    train_loader, val_loader, test_loader = dataloaders
    for epoch in range(args.epochs):
        print(epoch)
        count = 0
        train_loss, val_loss = 0,0
        train_prec = torch.zeros(1, args.n_classes, dtype=torch.float)
        val_prec = torch.zeros(1, args.n_classes, dtype=torch.float)
    
        for (images, images_body, labels_disc, labels_cont) in train_loader:
            if (count == 1):
                break
            images = images.to(dev)
            labels_disc = labels_disc.to(dev)
            images_body = images_body.to(dev)
            labels_cont = labels_cont.to(dev) 
            
            if (args.model_type == 'I'):
                out_category, out_cont = model(images)
            elif (args.model_type == 'B'):
                out_category, out_cont = model(images_body)
            elif (args.model_type == 'BI'):
                out_category, out_cont = model(images,images_body)

            labels_cont = labels_cont/10
           
            if (args.option=='joint'):
                loss_disc = criterion_disc(out_category, labels_disc.type(torch.FloatTensor).to(dev))
                loss_cont = criterion_cont(out_cont, labels_cont.type(torch.FloatTensor).to(dev))
                loss = args.w_disc * loss_disc + args.w_cont * loss_cont
                
            elif (args.option=='disc'):
                loss = criterion_disc(out_category, labels_disc.type(torch.FloatTensor).to(dev))
                
            print(loss.item())
            
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_prec +=  cal_precision(out_category,labels_disc,args)
            
            count += 1

        else:
            with torch.no_grad():
                model.eval()
                for (images, images_body, labels_disc, labels_cont) in val_loader:
                    images = images.to(dev)
                    labels_disc = labels_disc.to(dev)
                    images_body = images_body.to(dev)
                    labels_cont = labels_cont.to(dev)

                    if (args.model_type == 'I'):
                        val_out_category, val_out_cont = model(images)
                    elif (args.model_type == 'B'):
                        val_out_category, val_out_cont = model(images_body)
                    elif (args.model_type == 'BI'):
                        val_out_category, val_out_cont = model(images,images_body)

                    labels_cont = labels_cont/10
                
                    if (args.option == 'joint'):
                        loss_disc = criterion_disc(val_out_category, labels_disc.type(torch.FloatTensor).to(dev))
                        loss_cont = criterion_cont(val_out_cont, labels_cont.type(torch.FloatTensor).to(dev))
                        loss = args.w_disc * loss_disc + args.w_cont * loss_cont
                    elif (args.option == 'disc'):
                        loss = criterion_disc(val_out_category, labels_disc.type(torch.FloatTensor).to(dev))
                        
                    print(loss.item())
                    val_loss += loss.item()
                    
                    val_prec += cal_precision(val_out_category,labels_disc,args)
            model.train()
        
        print("Epoch: {}/{}.. ".format(epoch, args.epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader)))

        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        train_precision.append(train_prec/len(train_loader))
        val_precision.append(val_prec/len(val_loader))
    
    checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, 'emotion_model.pth')
    #evaluate(model, test_loader, args)
    tr = class_wise_precision(train_precision, args)
    val = class_wise_precision(val_precision, args)
    show_featuremap(train_loader,args,dev)
    torch.save(tr, 'train_prec.pt')
    torch.save(val, 'val_prec.pt')
    
def evaluate(model, test_loader, args):
    test_prec = torch.zeros(1, args.n_classes, dtype=torch.float)
    test_error_rate = torch.zeros(1,3, dtype=torch.float)
    
    for i, data in enumerate(test_loader):
        images, images_body, labels_disc, labels_cont = data
        images, labels_disc = images.to(dev), labels_disc.to(dev)
        images_body,labels_cont = images_body.to(dev), labels_cont.to(dev)
        
        
        labels_cont = labels_cont/10

        with torch.no_grad():
            if (args.model_type == 'I'):
                out_category,out_cont = model(images)
            elif (args.model_type == 'B'):
                out_category,out_cont = model(images_body)
            elif (args.model_type == 'BI'):
                out_category,out_cont = model(images, images_body)

            test_prec += cal_precision(out_category,labels_disc,args)
            test_error_rate += cal_error_rate(out_cont,labels_cont.type(torch.FloatTensor),args)
            average_precision(out_category,labels_disc,args)

    print('test precision')
    print(test_prec/len(test_loader))
    print('Mean error rate')
    print(test_error_rate/len(test_loader))
    
    return

