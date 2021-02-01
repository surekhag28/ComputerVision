import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

class ImageRotate:
    
    img = Image.open('./images/face_01_u6724013.jpg')
    img = img.resize((512,512))
    img = np.asarray(img)
    
    width,height,c = img.shape
    
    def rotateBackward(self,angle):
        angle = angle
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
        backwardMap = np.zeros((rowsf,colsf,3),dtype="uint8")
        
        # calculating center of new image
        cX = math.ceil((backwardMap.shape[0])/2)
        cY = math.ceil((backwardMap.shape[1])/2)

        for i in range(backwardMap.shape[0]):
            for j in range(backwardMap.shape[1]):
                xb = (i-cX)*math.cos(radians)-(j-cY)*math.sin(radians)
                yb = (i-cX)*math.sin(radians)+(j-cY)*math.cos(radians)
                xb = int(np.round(xb)+x)
                yb = int(np.round(yb)+y)
                if ((xb>=1 and xb<self.width) and (yb>=1 and yb<self.height)):
                    backwardMap[i,j,0]=self.img[xb,yb,0]  
                    backwardMap[i,j,1]=self.img[xb,yb,1]  
                    backwardMap[i,j,2]=self.img[xb,yb,2]  
          
        axs[1].imshow(backwardMap)
        axs[1].set_title("Backward Mapping ->> "+str(angle)+" angle")
        
def main():
    angles = [-90,-45,-15,45,90]
    for i in range(len(angles)):
        ImageRotate.rotateBackward(ImageRotate,angles[i])
    
if __name__ == '__main__':
    main()