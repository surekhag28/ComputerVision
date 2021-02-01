"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Surekha Gaikwad (Uni ID:- U6724013)
"""

import numpy as np


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    print(shape)
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import cv2

#bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image

filename = 'Harris_3.jpg'
img = cv2.imread(filename, cv2.IMREAD_COLOR)

if len(img.shape) > 2:
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################
k = 0.04

# parameters for marking the corners on image
radius = 1
color = (0, 0, 255)  # Green
thickness = 2

R = np.zeros((bw.shape[0],bw.shape[1]),np.float32)

# computes harris cornerness for the image
def harris_corners(bw):
    w, h = bw.shape
    
    # For every pixel find the corner strength or cornerness and store in array "R"
    for i in range(w):
        for j in range(h):
            # computing M matrix by taking corresponding values from all the smoothed covariance matrices
            M = np.array([[Ix2[i][j], Ixy[i][j]], [Ixy[i][j], Iy2[i][j]]])
            
            # computing the corner strenth value using determinant and trace of M matrix
            R[i][j] = np.linalg.det(M) - (k * np.square(np.trace(M)))
    
    return R
        
######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
# Plot detected corners on image
        
# performing non max supression to find local maxima and supress non-local maxima
def non_max_supression(bw,R,window = 5):
    # performing thresholding to select values which are greater than threshold and remaining as 0.
    R=R*(R>(thresh*R.max()))
    
    final_scores=np.zeros(R.shape,dtype=np.float32)
    r,c=np.nonzero(R)

    # to compute the local maxima within certain window range or distance and ignoring the non maxima
    for row,column in zip(r,c):
        # computing the coordinates of window range from which we will consider corner values
        min_row=max(0,row-window//2)
        max_row=min(bw.shape[0],min_row+window)
        min_column=max(0,column-window//2)
        max_column=min(bw.shape[1],min_column+window)
        
        # to check if the current corner value is greater than the max value within window
        if (R[row,column]==R[min_row:max_row,min_column:max_column].max()):
            # matrix storing coordinates of corner point which has maximum corner value
            final_scores[row,column]=R[row,column] 
            cv2.circle(img, (column, row), radius, color, thickness)

    cv2.imwrite('./results/'+filename +'_output.jpg', img)
    return final_scores


R = harris_corners(bw)
final_scores = non_max_supression(bw,R,window = 5)

######################################################################
# Now performing the harris corner detection using inbuilt opencv method 
#       and comparing the results with previously coded method.
######################################################################


img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite('./results/'+filename+'_opencv.jpg', img)