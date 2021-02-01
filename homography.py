#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:42:29 2020

@author: surekhagaikwad
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inp = 6


# Normalization of coordinates.
def Normalization(nd, x):
    '''
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = np.linalg.inv(Tr)
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:nd, :].T

    return Tr, x

def getArray(X,Y):
    
    arr = np.zeros((6,2))
    
    for i in range(6):
        arr[i][0] = X[i]
        arr[i][1] = Y[i]
        
    return arr
    
# functiom to compute homography matrix given points correspondences
def computeHomography(u2Trans,v2Trans, uBase,vBase):
    H = np.eye(3)
    
    uvTrans = getArray(u2Trans,v2Trans)
    uvBase = getArray(uBase,vBase)
    
    # Normalize the data to improve the DLT quality.
    Txyz, xyzn = Normalization(2, uvTrans)
    Tuv, uvn = Normalization(2, uvBase)
    
    A = []
    for i in range(u2Trans.shape[0]):
        # image points corresponding to base image i.e. left image 
        x1, y1 = uvn[i, 0], uvn[i, 1]
        
        # image points corresponding to transformed image i.e. right image
        x2, y2 = xyzn[i, 0], xyzn[i, 1]

        # creating matrix A by stacking up all the point corespondences, so for 6 points
        # there will 12 entries in matrix A.
        
        A.append([-x1,-y1,-1,0,0,0,x1*x2,y1*x2,x2])
        A.append([0,0,0,-x1,-y1,-1,x1*y2,y1*y2,y2])
    
    A = np.array(A)

    # performing SVD to compute h vector.
    U,S,V = np.linalg.svd(A)
    
     # The parameters are in the last line of V and normalize them
    L = V[-1, :] / V[-1, -1]
    
    H = L.reshape(3, 3)
    
    # Denormalization
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    H = H / H[-1, -1]
    
    return H

def getData(file,x,y):
    df = pd.read_excel(file)
    for i in range(inp):
        x[i] = df.iat[i,0]
    for j in range(inp):
        y[j] = df.iat[j,1]

    return x,y

u2Trans = np.zeros(inp)
v2Trans = np.zeros(inp)
uBase = np.zeros(inp)
vBase = np.zeros(inp)


uBase,vBase = getData('left.xlsx',uBase,vBase)
u2Trans,v2Trans = getData('right.xlsx',u2Trans,v2Trans)
    
img1 =  cv2.imread("Left.jpg");
img2 =  cv2.imread("Right.jpg");

H = computeHomography(u2Trans,v2Trans, uBase,vBase)
print(H)

sz = img2.shape
warp_img = cv2.warpPerspective(img1, H, (sz[1],sz[0]))


fig,axs = plt.subplots(1,3,figsize=(20,15))
axs[0].imshow(img1)
axs[0].set_title("Left Image")
axs[0].axis("off")

axs[1].imshow(img2)
axs[1].set_title("Right Image")
axs[1].axis("off")


axs[2].imshow(warp_img)
axs[2].set_title("Warped Left Image")
axs[2].axis("off")

plt.show()