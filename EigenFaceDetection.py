#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:44:24 2020

@author: surekhagaikwad
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

width  = 195
height = 231
train_path = './Yale-FaceA/trainingset/'
test_path = './Yale-FaceA/testset/'

# to read images from directory
def read_data(data_path):
    data = []
    for path in glob.glob(data_path+'*.png'):
        f_image = Image.open(path).convert('L')
        data.append(np.asarray(f_image,dtype=float)/255.0)
    data = np.asarray(data)
    
    return data
        
 
# to display mean/average face for trainingset
def display_mean_face(train_data):
    mean = np.mean(train_data,0)
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    axs.imshow(mean, cmap=plt.cm.gray)
    axs.set_title("Mean Face")
    plt.show()
    return mean
    
# to display top 10 eigne faces
def display_eigen_faces(eigen_faces):
    fig, axs = plt.subplots(2, 5)
    fig.set_size_inches(10, 10)
    count = 0
    for x in range(2):
        for y in range(5):
            draw_image = eigen_faces[count]
            axs[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axs[x][y].axis('off')
            count = count + 1
    fig.suptitle("Top 10 Eigen Faces")
    plt.show()
  
# to display reconstructed faces using top 10 eignevectors/eignefaces
def display_reconstruct_faces(images):
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(10, 5)
    count = 0
    for x in range(5):
        draw_image = np.reshape(images[x,:],(height,width))
        axs[x].imshow(draw_image, cmap=plt.cm.gray)
        axs[x].axis('off')
        count = count + 1
    fig.suptitle("Reconstructed Faces using 10 eigenvectors.")
    plt.show()
  
# to compute eigenvectors or eigenfaces from training data
def compute_pca(train_data,mean_face):
    train_vector = []
    for i in range(len(train_data)):
        train_vector.append(train_data[i].flatten())
    train_vector = np.asarray(train_vector)
    # compute mean face using training data
    mean = mean_face.flatten()
    # normalize the training data using mean face
    normalized_train = np.subtract(train_vector,mean)
    
    # perform singular value decomposition on normalized data
    U, s, V = np.linalg.svd(normalized_train,full_matrices=False)
    plt.plot(s)
    eigen_faces=[]
    
    # display top 10 eigen faces
    for x in range(10):
        image=np.reshape(V[x],(height,width))
        eigen_faces.append(image)
    display_eigen_faces(eigen_faces)
    return V,normalized_train
   
# to recconstruct faces using pca components
def reconstruct_faces(mean_face,normalized_train,V):
    # get the weights of training data by projecting it on eigenvectors
    weights=np.dot(normalized_train, V.T)
    reconstruction = mean_face + np.dot(weights[:,0:10], V[0:10,:])
    display_reconstruct_faces(reconstruction)
   
# to find top 3 matched faces from training data for each of the test face
def recognize_face(normalized_test,V,normalized_train,train_data,test_data):
    # get the weights of training data by projecting it on pca components
    train_weights = np.dot(V[:10, :],normalized_train.T)
    correct_pred = 0
    threshold = 1000
    for i in range(normalized_test.shape[0]):
        # get the weights of test data by projecting it on pca components
        test_weights = np.dot(V[:10, :],normalized_test[i:i + 1,:].T)
        
        # compute euclidean distance between training and testing weights to find which face matches
        # the test face
        euclidean_distance = np.sum((train_weights - test_weights) ** 2, axis=0)
        
        sort_distance = np.sort(euclidean_distance)
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(10, 5)
        img=np.reshape(normalized_test[i,:], (height,width))
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].axis('off')
        axs[0].set_title("Test Face")
        
        count = 0
        
        # to plot the top 3 matched faces from training data for each of the test face
        for j in range(3):
            for k in range(len(euclidean_distance)):
                if(sort_distance[j]==euclidean_distance[k]):
                    axs[j+1].imshow(train_data[k,:,:], cmap=plt.cm.gray)
                    axs[j+1].axis('off')
                    axs[j+1].set_title("Matched Face")
                    
                    print(sort_distance[j],k)
                    
                    if(sort_distance[j]<=threshold):
                        
                        count += 1
                    break
        plt.show()
        if(count == 3):
            correct_pred += 1
        
    num_images = normalized_test.shape[0]
    print('Accuracy: {}/{} = {}%'.format(correct_pred-1, num_images, (correct_pred-1)/num_images*100.00))
            

train_data = read_data(train_path)
mean_face = display_mean_face(train_data)
V,normalized_train = compute_pca(train_data,mean_face.flatten())
reconstruct_faces(mean_face.flatten(),normalized_train,V)


test_data = read_data(test_path)
test_vector = []
for i in range(len(test_data)):
    test_vector.append(test_data[i].flatten())
test_data = np.asarray(test_vector)

normalized_test = np.subtract(test_data,mean_face.flatten())
recognize_face(normalized_test,V,normalized_train,train_data,test_data)

