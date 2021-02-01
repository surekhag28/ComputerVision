import math
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class ImageRotate:
    img = Image.open('./images/face_01_u6724013.jpg')
    img = img.resize((512,512))
    img = np.asarray(img)

    #img = img.resize((512,512))
    width,height,c = img.shape
    
    def rotateForward(self,angle):
        angle = angle
        #calculating angle in radians
        radians = float(angle*(math.pi/180))
        
        fig,axs = plt.subplots(1,2,figsize=(10,7))

        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        
        # calculating the size of new image whose shape will change when its rotated
        rowsf=math.ceil(self.width*abs(math.cos(radians))+self.height*abs(math.sin(radians)))                     
        colsf=math.ceil(self.width*abs(math.sin(radians))+self.height*abs(math.cos(radians)))

        # calculating center of original image
        x=math.ceil(self.width/2)                                                          
        y=math.ceil(self.height/2)
        
        # creating new image with new width and height
        forwardMap = np.zeros((rowsf,colsf,3),dtype="uint8")
        
        # calculating center of new image
        cX = math.ceil((forwardMap.shape[0])/2)
        cY = math.ceil((forwardMap.shape[1])/2)
        
        for i in range(self.width):
            for j in range(self.height):
                # calculating new coordinates of rotating image
                xf = (i - x) * math.cos(radians) + (j - y) * math.sin(radians)
                yf = -(i - x) * math.sin(radians) + (j - y) * math.cos(radians) 
                
                # updating the coordinates since rotating image aorund center.
                xf = int(np.round(xf)+cX)
                yf = int(np.round(yf)+cY)
                
                # finally setting pixel value from original image to new location in rotating image.
                if xf in range(forwardMap.shape[0]) and yf in range(forwardMap.shape[1]):
                    #forwardMap[xf, yf] = self.img[i, j]
                    forwardMap[xf, yf,0] = self.img[i,j,0]
                    forwardMap[xf, yf,1] = self.img[i,j,1]
                    forwardMap[xf, yf,2] = self.img[i,j,2]
                else:
                    pass
        
        axs[1].imshow(forwardMap)
        axs[1].set_title("Forward Mapping ->> "+str(angle)+" angle")
        
def main():
    angles = [-90,-45,-15,45,90]
    for i in range(len(angles)):
        ImageRotate.rotateForward(ImageRotate,angles[i])

if __name__ == '__main__':
    main()