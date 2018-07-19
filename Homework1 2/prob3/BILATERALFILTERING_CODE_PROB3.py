import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import imageio
from PIL import Image
import tifffile as tiff
import math

##########distance function#######################
def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)
#################################

########## gaussain function#######################
def gaussian(x, sigma):
    return (1.0 / np.sqrt((2 * math.pi * (sigma ** 2)))) * np.exp(- (x ** 2) / (2 * sigma ** 2))
#################################

########## section 3.b <bilateral filtering function >#######################
def bilateral_filter_own(source, x, y, diameter, sigma_i, sigma_s):
    #print(type(source))
    hl = int(diameter/2)
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    return (i_filtered / Wp)
###################################################

########## section 3.b <calling bilateral filtering function with following parameters>#######################
if __name__ == "__main__":
    imh= cv2.imread('babyelephant.jpg')
    diam = 5
    ele_new = np.zeros([imh.shape[0], imh.shape[1],3])    
    for c in range(0,3):
        #print(src.shape[0])
        #print(src.shape[1])
        for l in range(0, imh.shape[0]):
            for m in range(0, imh.shape[1]):
            
                imh[l,m,c] = bilateral_filter_own(imh[:,:,c],l,m,diam,15,30)

cv2.imwrite('bilateral_bl_elephant.jpg', np.uint8(imh))
################################################################

########## section 3.a <Loading and displaying the image >#######################
#load image
img = cv2.imread('babyelephant.jpg')
#display image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
####################################

########## section 3.a <displaying the image after applying gaussian blur >#######################
#display image
dst = cv2.GaussianBlur(img, (5,5),0)
cv2.imshow('gaussian blurred image',dst) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("gaussianblurred.jpg", dst)
####################################

###################3.b comparision of original, gaussian blurred and bilateral images#######################
img = cv2.imread('original_image_grayscale.png')
img1 = cv2.imread('gaussianblurred.jpg')
img2 = cv2.imread('bilateral_bl_elephant.jpg')

fig = plt.figure()

a=fig.add_subplot(2,3,1)
az = img[180:250,180:250]
imgk = plt.imshow(az)
a.set_title('original')
plt.colorbar(orientation='horizontal')

a1=fig.add_subplot(2,3,2)
az1 = img1[180:250,180:250]
imgk = plt.imshow(az1)
a1.set_title('gaussian blurred')
plt.colorbar(orientation='horizontal')

a2=fig.add_subplot(2,3,3)
az2 = img2[180:250,180:250]
imgk = plt.imshow(az2)
a2.set_title('bilinear')
plt.colorbar(orientation='horizontal')

a3=fig.add_subplot(2,3,4)
az3 = img[250:350,250:350]
imgk = plt.imshow(az3)
a3.set_title('original')
plt.colorbar(orientation='horizontal')

a4=fig.add_subplot(2,3,5)
az4 = img1[250:350,250:350]
imgk = plt.imshow(az4)
a4.set_title('gaussian blurred')
plt.colorbar(orientation='horizontal')

a5=fig.add_subplot(2,3,6)
az5 = img2[250:350,250:350]
imgk = plt.imshow(az5)
a5.set_title('bilinear')
plt.colorbar(orientation='horizontal')

plt.show()
####################################################################