#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:31:39 2019

@author: surekhagaikwad
"""
import argparse
import torch
from torchvision import transforms
from EmotionDataset import EmotionDataset
from model import CombinedModel_BI, Model_I, Model_B
from train import train

dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(dev)

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--w_disc', type=int, default=1/2)
    parser.add_argument('--w_cont', type=int, default=1/2)
    parser.add_argument('--n_classes', type=int, default=26)
    parser.add_argument('--model_type', type=str, default='BI', help='It can be only Image, only Body or both')
    parser.add_argument('--option', type=str, default='joint', help='for deciding loss function')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    
    transform = transforms.Compose([transforms.Resize([227,227]), #256 # for alexnet 227
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])

    test_transform = transforms.Compose([transforms.Resize([227,227]),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])

    train_data = EmotionDataset(disc_file = '../data/train_labels.csv', cont_file = '../data/train_labels_cont.csv',
                                        root_dir = '../data/train/', flag='train', transform = transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)

    val_data = EmotionDataset(disc_file = '../data/val_labels.csv', cont_file = '../data/val_labels_cont.csv',
                                    root_dir = '../data/val/', flag='val', transform = transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=1)

    test_data = EmotionDataset(disc_file = '../data/test_labels.csv', cont_file = '../data/test_labels_cont.csv',
                                    root_dir = '../data/test/', flag='test', transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=1)

    
    dataloaders = (train_loader, val_loader, test_loader)

     
    if (args.model_type == 'BI'):
        model = CombinedModel_BI(args).to(dev)
    elif (args.model_type == 'I'):
        model = Model_I(args).to(dev)
    elif (args.model_type == 'B'):
        model = Model_B(args).to(dev)
        
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
   
    train(args, model, optimizer, dataloaders)
    print('training finished')
