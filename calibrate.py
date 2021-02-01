# -*- coding: utf-8 -*-
# CLAB3 
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.linalg import rq
from numpy.linalg import inv

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

'''This function computes the decomposition of the projection matrix into intrinsic 
parameters, K, and extrinsic parameters Q (the rotation matrix) and t 
(the translation vector) '''

def P_to_KRt(P):
  
  M = P[0:3,0:3]
  
  R, Q = rq(M)
    
  K = R/float(R[2,2])
  
  if K[0,0] < 0:
    K[:,0] = -1*K[:,0]
    Q[0,:] = -1*Q[0,:]
    
  if K[1,1] < 0:
    K[:,1] = -1*K[:,1]
    Q[1,:] = -1*Q[1,:]
  
  P_1 = np.dot(K,Q)
  
  P_scale = (P_1[0,0]*P)/float(P[0,0])
  
  t = np.dot(inv(K), P_scale[:,3])
  
  return K, Q, t

def display2DPoints(uv,uv2):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    x=uv[:,0]
    y=uv[:,1]
    x_proj=uv2[:,0]
    y_proj=uv2[:,1]
    for xx,yy in zip(x,y):
        circ = Circle((xx,yy),10,color='r')
        ax.add_patch(circ)
        
    for xx,yy in zip(x_proj,y_proj):
        circ = Circle((xx,yy),5,color='g')
        ax.add_patch(circ)
        
    

    ax.imshow(I) 
    plt.title("Original and projected points on image")
    plt.axis("off")
    plt.show()
   
#Camera calibration by DLT using known world coordinates and their image points.
def calibrate(nd, xyz, uv):
    
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Normalize the data to improve the DLT quality.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    A = np.asarray(A) 

    # Find 12 parameters using SVD
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of V and normalize them
    L = V[-1, :] / V[-1, -1]
    
    # Camera projection matrix
    H = L.reshape(3, nd + 1)

    # Denormalization
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    H = H / H[-1, -1]
    C = H.flatten(order='C')

# Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates)
    uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    
    # mean squared error distance
    mean_err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 
    
    K,R,t = P_to_KRt(H)
    print(K.shape,R.shape,t.shape)
    
    uv2 = np.zeros((n, 2))
    
    # for displaying projected 3D points on image 
    dist_coeff = np.zeros((4, 1))
    
    # computing the projected image points using camera matrix and world coordinates 
    for i in range(n):
        x = np.array(xyz[i], dtype=np.float64)
        proj, _ = cv2.projectPoints(x, R, t, K, dist_coeff)
        uv2[i,0] = proj[0][0][0]
        uv2[i,1] = proj[0][0][1]
    
    # for displaying original image points and projected world points.
    display2DPoints(uv,uv2)
    

    return C,K,R, mean_err

    

'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% Surekha Gaikwad(U6724013), 12/05/2020 
'''



############################################################################
def getArray(X,Y):
    
    arr = np.zeros((6,2))
    
    for i in range(6):
        arr[i][0] = X[i]
        arr[i][1] = Y[i]
        
    return arr

def homography(u2Trans, v2Trans, uBase, vBase):
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

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% Surekha Gaikwad(U6724013), 12/05/2020 
'''

def getData(file,x,y):
    df = pd.read_excel(file)
    for i in range(inp):
        x[i] = df.iat[i,0]
    for j in range(inp):
        y[j] = df.iat[j,1]

    return x,y

if __name__ == "__main__":
    I = Image.open('stereo2012a.jpg');
    
    df = pd.DataFrame()
    df = pd.read_excel('image_points.xlsx')
    uv = df.to_numpy()
    
    df = pd.read_excel('world_points.xlsx')
    xyz = df.to_numpy()

    nd = 3
    P,K,R,mean_err = calibrate(nd, xyz, uv)
    print("Camera Matrix:-")
    print(P)
    print("\nCamera Intrinsic Matrix:-")
    print(K)
    print("\nRotation Matrix:-")
    print(R)
    print("\nError:-")
    print(mean_err)
    import math
    
    P = P.reshape(3, nd + 1)
    
    euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
    euler_angles_degrees = 180 * euler_angles_radians/math.pi
    
    eul    = euler_angles_radians
    yaw    = 180*eul[1,0]/math.pi # warn: singularity if camera is facing perfectly upward. Value 0 yaw is given by the Y-axis of the world frame.
    pitch  = 180*((eul[0,0]+math.pi/2)*math.cos(eul[1,0]))/math.pi
    
    
    inp = 6
    u2Trans = np.zeros(inp)
    v2Trans = np.zeros(inp)
    uBase = np.zeros(inp)
    vBase = np.zeros(inp)


    uBase,vBase = getData('left.xlsx',uBase,vBase)
    u2Trans,v2Trans = getData('right.xlsx',u2Trans,v2Trans)
        
    img1 =  cv2.imread("Left.jpg");
    img2 =  cv2.imread("Right.jpg");
    
    H = homography(u2Trans,v2Trans, uBase,vBase)
    print("\n\nHomography Matrix:- \n",H)
    
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