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
from scipy.linalg import svd,lstsq
from warp_image import warpImage

inp = 6

# functiom to compute homography matrix given points correspondences
def computeHomography(u2Trans,v2Trans, uBase,vBase):
    A = []
    b = []
    
    for i in range(u2Trans.shape[0]):
        # image points corresponding to base image i.e. left image
        x1 = uBase[i]
        y1 = vBase[i]
        
        # image points corresponding to transformed image i.e. right imag
        x2 = u2Trans[i]
        y2 = v2Trans[i]

        # creating matrix A by stacking up all the point corespondences, so for 6 points
        # there will be 12 entries in matrix A.
        A.append([-x1,-y1,-1,0,0,0,x1*x2,y1*x2])
        A.append([0,0,0,-x1,-y1,-1,x1*y2,y1*y2])
        
        
        b.append([x2])
        b.append([y2])
    
    A = np.array(A)
    b = np.array(b)

    h = lstsq(A,b)
    #h = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
    H = np.reshape(np.vstack((h[0],[1])),(3,3))  
    
    #H = (1/H.item(8)) * H
    
    #H = np.matmul(np.linalg.pinv(A), b)

    #H = np.reshape(np.vstack((h,[1])),(3,3)) 
    return H

def getData(file,x,y):
    df = pd.read_excel(file)
    for i in range(inp):
        x[i] = df.iat[i,0]
    for j in range(inp):
        y[j] = df.iat[j,1]

    return x,y

def interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 2)
    fy = round(new_y - int(new_y), 2)

    p = np.zeros((3,))
    p += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)]
    p += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
    p += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
    p += fx * fy * img[int(new_y) + 1, int(new_x) + 1]

    return p
def backward_warpping(img, output,H):
    h, w, ch = output.shape
    homography_matrix = H
    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            res = interpolation(img, new_x, new_y)
            output[y][x] = res

    return output


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

output = np.zeros((sz[0], sz[1], 3))

#warp_img = backward_warpping(img1, output, H)
#warp_img = warp_img.astype(np.uint8)


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