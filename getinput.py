#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:42:29 2020

@author: surekhagaikwad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
  


img = cv2.imread('Left.jpg')

fig,axs = plt.subplots(1,1,figsize=(10,7))

#result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

axs.imshow(img)
axs.set_title("Clustered Image")
axs.axis("off")
    
print("After 6 clicks :") 
x = plt.ginput(2) 
print(x) 
   
plt.show() 
plt.close()

p1 = np.zeros((2,2))
for i in range(2):
    p1[i,0] = x[0][0]
    p1[i,1] = x[0][1]
    
print(p1)
    
    