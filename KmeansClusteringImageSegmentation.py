#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:32:26 2020

@author: surekhagaikwad
"""

import cv2
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from random import randint as randi
import matplotlib.pyplot as plt

# read original image and then create new image of shape (rows*cols) x 5
# each data point or pixel is represented as 5D vector
def get_data(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    print(img.shape)
    
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    
    new_img = np.empty(((img.shape[0]*img.shape[1]),5))
            
    k = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[k,0] = img[i,j,0]
            new_img[k,1] = img[i,j,1]
            new_img[k,2] = img[i,j,2]
            new_img[k,3] = i
            new_img[k,4] = j
            k += 1
    
    return new_img,img

# compute initial centroids radomly
def get_centroids(centroids,n_clusters,img):
    
    for i in range(n_clusters):
        x = randi(0, img.shape[0]-1)
        y = randi(0, img.shape[1]-1)
        l = float(img[x, y, 0])
        a = float(img[x, y, 1])
        b = float(img[x, y, 2])
        
        centroids[i,0] = l
        centroids[i,1] = a
        centroids[i,2] = b
        centroids[i,3] = x
        centroids[i,4] = y
        
    return centroids
        
# computing eudlidean distance of each data point from centroid
def get_euclidean_distance(centroid,new_img):
    l = float(centroid[0] - new_img[0])
    a = float(centroid[1] - new_img[1])
    b = float(centroid[2] - new_img[2])
    x = float(centroid[3] - new_img[3])
    y = float(centroid[4] - new_img[4])
    
    dist = np.sqrt(l*l + a*a + b*b + x*x + y*y)
    
    return dist

# computing new centroid after every iteration    
def compute_new_centroids(centroids,c_labels,new_img):
    
    new_values = np.zeros((centroids.shape[0],6), dtype=np.float64)
    for i in range(new_img.shape[0]):
        
        new_values[c_labels[i,0],0] += 1
        new_values[c_labels[i,0],1] += float(new_img[i,0])
        new_values[c_labels[i,0],2] += float(new_img[i,1])
        new_values[c_labels[i,0],3] += float(new_img[i,2])
        new_values[c_labels[i,0],4] += float(new_img[i,3])
        new_values[c_labels[i,0],5] += float(new_img[i,4])
        
    for j in range(centroids.shape[0]):
        centroids[j,0] = new_values[j,1]/new_values[j,0]
        centroids[j,1] = new_values[j,2]/new_values[j,0]
        centroids[j,2] = new_values[j,3]/new_values[j,0]
        centroids[j,3] = new_values[j,4]/new_values[j,0]
        centroids[j,4] = new_values[j,5]/new_values[j,0]
    
    return centroids

# displaying the final clusters of image
def display_cluster(clusterLabels,centroids,img):
    result = np.zeros((new_img.shape[0],3), dtype=np.uint8)
    for i in range(new_img.shape[0]):
        lab = centroids[clusterLabels[i]]
        result[i,0] = np.uint8(lab[0,0])
        result[i,1] = np.uint8(lab[0,1])
        result[i,2] = np.uint8(lab[0,2])

    result = np.reshape(result,(img.shape[0],img.shape[1],3))
    result = cv2.resize(result,None,fx=3,fy=3)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    cv2.imwrite('segmented_1.png', result)
    
    fig,axs = plt.subplots(1,2,figsize=(10,7))

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(result)
    axs[1].set_title("Clustered Image")
    axs[1].axis("off")
    plt.show()
        
# kmeans algorithm to assign category labels to each data point depending on their distance
# from centroids
def my_kmeans(img,new_img,n_clusters=6):
    max_iteration = 100
    
    centroids = np.zeros((n_clusters, 5), dtype=np.float64)
    centroids = get_centroids(centroids,n_clusters,img)
    
    c_labels = np.zeros((new_img.shape[0],1), dtype=np.uint8)
    for i in range(max_iteration):
        print(i)
        for j in range(new_img.shape[0]):
            min_dist = sys.float_info.max
            for k in range(n_clusters):
                euclidean_dist = get_euclidean_distance(centroids[k],new_img[j])
                if(euclidean_dist <= min_dist):
                    min_dist = euclidean_dist
                    # data point is assigned label if it is near to selected centroid
                    c_labels[j,0] = k
                    
        old_centroids = np.copy(centroids) 
        centroids = compute_new_centroids(centroids,c_labels,new_img)
        
        flag = True
        
        # condition check to decide when to stop iteration
        for x in range(centroids.shape[0]):
            for y in range(centroids.shape[1]):
                if(old_centroids[x,y]!=centroids[x,y]):
                    flag = False
                    break
            if(flag==False):
                break
        if(flag==True):
            break
        
    display_cluster(c_labels,centroids,img)
                    
file_names = ['peppers.png','mandm.png']
for f in file_names:      
    new_img,img = get_data(f)
    clusters = [8,12,25]
    for c in clusters:
        start_time = time.time()        
        my_kmeans(img,new_img,n_clusters=c)
        print("--- %s seconds ---" % (time.time() - start_time))