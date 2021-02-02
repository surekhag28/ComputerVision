#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:57:47 2019

@author: surekhagaikwad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:38:44 2019

@author: surekhagaikwad
"""

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.imageFeature_conv, target_layers)
        self.max_pool = self.model.image_max_pool
        self.category = self.model.category
        
        self.cont = self.model.cont
        self.sigmoid = self.model.sigmoid
    
    def get_gradients(self):
        return self.feature_extractor.gradients
    
    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = self.max_pool(output)
        
        output = output.view(-1,256*6*6)
        
        output = self.model.fusion(output)
        out_category = self.sigmoid(self.category(output)) 
        out_cont = self.sigmoid(self.cont(output))
        
        return target_activations, out_category, out_cont

def preprocess_image(img):
    means=[0.5, 0.5, 0.5]
    stds=[0.5, 0.5, 0.5]
    
    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)
        
    def forward(self, input):
        return self.model(input) 
   
    def __call__(self, input, index = None):
        if self.cuda:
            features, output, output_cont = self.extractor(input.cuda())
        else:
            features, output, output_cont = self.extractor(input)
        
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
           one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.model.imageFeature_conv.zero_grad()
        self.model.fusion.zero_grad()
        one_hot.backward()
        
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
    
        for i, w in enumerate(weights):	
            cam += w * target[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (227, 227))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):	
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()		
        
        for idx, module in self.model.imageFeature_conv._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.imageFeature_conv._modules[idx] = GuidedBackpropReLU()
    
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, index = None):
        if self.cuda:
            out_category,out_cont = self.forward(input.cuda())
        else:
            out_category,out_cont = self.forward(input)
            
        if index == None:
            index = np.argmax(out_category.to('cpu').data.numpy())
            
        one_hot = np.zeros((1, out_category.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * out_category)
        else:
            one_hot = torch.sum(one_hot * out_category)
        
        one_hot.backward()
        
        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]
        
        return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='../data/train/4flffefotbl974fx76.jpg',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model

if __name__ == '__main__':
    args = get_args()
    model = load_checkpoint('emotion_model.pth')
    
    grad_cam = GradCam(model = model, \
					target_layer_names = ["10"], use_cuda=args.use_cuda)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (227, 227))) / 255
    input = preprocess_image(img)
    target_index = None
    mask = grad_cam(input, target_index)
    show_cam_on_image(img, mask)
    gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), 'gb.jpg')
    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
	    cam_mask[i, :, :] = mask
    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')