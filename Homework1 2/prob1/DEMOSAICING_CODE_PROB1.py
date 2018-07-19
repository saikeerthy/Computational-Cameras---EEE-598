import rawpy
import imageio
import cv2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
#########1a.reading and saving the image in png format and subsampling ##############################################
path = 'tetons.nef'
raw = rawpy.imread(path)
rgb = raw.postprocess()
imageio.imsave('tetons_original.png', rgb)

k=cv2.imread('tetons_original.png');
cvuint8 = cv2.convertScaleAbs(k)
print(cvuint8.dtype)

bayer = raw.raw_image
print(bayer.shape[1])
print(bayer.shape[0])

rr=bayer[::2, ::2] 
gg=(bayer[1::2, ::2]+bayer[::2, 1::2])*0.5  
bb=bayer[1::2, 1::2]

r1=(rr/float(np.max(rr)))*255
g1=(gg/float(np.max(gg)))*255
b1=(bb/float(np.max(bb)))*255

im=cv2.merge((b1,g1,r1))
cv2.imwrite("tetons_subsample.png",im)
print(im.shape[1])
print(im.shape[0])
##################################################################

#########1b. nearest neighbour demosaicing ##############################################
img_scaled2 = cv2.resize(im,None,fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
cv2.imwrite('tetons_nn.png', img_scaled2)
#####################################################################

#########1c. Bilinear demosaicing ##############################################
img_scaled3 = cv2.resize(im,None,fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
cv2.imwrite('tetons_bl.png', img_scaled3)
####################################################################

#########1d. Gunturk demosaicing ##############################################
rn=bayer[::2, ::2]
gn=np.zeros([int(bayer.shape[0]/2),int(bayer.shape[1]/2)])
bn=bayer[1::2, 1::2]

for x in range(0,bayer.shape[0],2):
    for y in range(0,bayer.shape[1],2):
        dh=np.abs(((bayer[x,0]+bayer[x,bayer.shape[1]-1])/2)-bayer[x,y])
        dv=np.abs(((bayer[0,y]+bayer[bayer.shape[0]-1,y])/2)-bayer[x,y])
        if dh>dv:
            gn[int(x/2),int(y/2)]=(bayer[x-1,y]+bayer[x+1,y])/2
        elif dh<dv:
            gn[int(x/2),int(y/2)]=(bayer[x,y-1]+bayer[x,y+1])/2
        else:
            gn[int(x/2),int(y/2)]=(bayer[x-1,y]+bayer[x,y-1]+bayer[x+1,y]+bayer[x,y+1])/4

rk=(rn/float(np.max(rn)))*255
gk=(gn/float(np.max(gn)))*255
bk=(bn/float(np.max(bn)))*255

imk = cv2.merge((bk,gk,rk))
cv2.imwrite('tetons_dm.png',imk)
#####################################################################

#########1e. Gunturk demosaicing ##############################################
img = cv2.imread('tetons_nn.png')
img1 = cv2.imread('tetons_bl.png')
img2 = cv2.imread('tetons_dm.png')

fig = plt.figure()

a=fig.add_subplot(2,3,1)
az = img[100:150,100:150]
imgk = plt.imshow(az)
a.set_title('nearest neighbour')
plt.colorbar(orientation='horizontal')

a1=fig.add_subplot(2,3,2)
az1 = img1[100:150,100:150]
imgk = plt.imshow(az1)
a1.set_title('bilinear')
plt.colorbar(orientation='horizontal')

a2=fig.add_subplot(2,3,3)
az2 = img2[100:150,100:150]
imgk = plt.imshow(az2)
a2.set_title('gunturk')
plt.colorbar(orientation='horizontal')

a3=fig.add_subplot(2,3,4)
az3 = img[150:200,150:200]
imgk = plt.imshow(az3)
a3.set_title('nearest neighbour')
plt.colorbar(orientation='horizontal')

a4=fig.add_subplot(2,3,5)
az4 = img1[150:200,150:200]
imgk = plt.imshow(az4)
a4.set_title('bilinear')
plt.colorbar(orientation='horizontal')

a5=fig.add_subplot(2,3,6)
az5 = img2[150:200,150:200]
imgk = plt.imshow(az5)
a5.set_title('gunturk')
plt.colorbar(orientation='horizontal')

plt.show()
#####################################################################