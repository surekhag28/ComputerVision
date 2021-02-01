import cv2
import os
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image

# Here we read the image and bring it as an array

GT_DIR = os.path.join('contour-data', 'groundTruth')
IMAGE_DIR = os.path.join('contour-data', 'images')

def get_imlist(name):
    imlist = np.loadtxt('contour-data/{}.imlist'.format(name))
    return imlist.astype(np.int)



def TransformKernel(kernel):
    transform_kernel = kernel.copy()    
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            transform_kernel[i][j] = kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    return transform_kernel

def GetPadddedImage(image):       
    imagePadded = np.asarray([[ 0 for x in range(0,image.shape[1] + 2)] for y in range(0,image.shape[0] + 2)], dtype =np.uint8)
    imagePadded[1:(imagePadded.shape[0]-1), 1:(imagePadded.shape[1]-1)] = image 
    return imagePadded 

def Convolution(image, kernel):    
    kernel = TransformKernel(kernel)    
    imagePadded = GetPadddedImage(image)           
    imageConvolution = np.zeros(image.shape)    
    for i in range(1, imagePadded.shape[0]-1):
        for j in range(1, imagePadded.shape[1]-1):
            sum = 0            
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sum += kernel[m][n]*imagePadded[i+m-1][j+n-1]
            imageConvolution[i-1][j-1] = sum                           
    list = []
    for i in range(imageConvolution.shape[0]):
        for j in range(imageConvolution.shape[1]):
            if(image[i][j] < 0):
                imageConvolution[i][j] = imageConvolution[i][j] * -1
            list.append(imageConvolution[i][j])          
    imageConvolution /= max(list)
    
    return imageConvolution


def GetEdgeMagnitute(img1, img2):    
    img_copy = np.zeros(img1.shape)    
    list = []
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q = (img1[i][j]**2 + img2[i][j]**2)**(1/2)            
            img_copy[i][j] = q
            list.append(q)
    img_copy /= max(list)
    return img_copy
imset = 'val'
imlist = get_imlist(imset)
output_dir = 'sobel_baseline/output/';
    
for imname in tqdm(imlist):
    original_image = cv2.imread(os.path.join(IMAGE_DIR, str(imname) + '.jpg'))
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)   

    out_file_name = os.path.join(output_dir, str(imname) + '.png')
    
    
    #edges = sobel_edge_detect(img_as_array)


    kernelY = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
    gradientY = Convolution(gray, kernelY)
    kernelX = np.asarray([[1,0,-1],[2,0,-2],[1,0,-1]])
    gradientX = Convolution(gray, kernelX)
    sobelEdgeMagnitute = GetEdgeMagnitute(gradientX,gradientY)

    img = sobelEdgeMagnitute * 255.
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    cv2.imwrite(out_file_name,img)
