import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import imageio
from PIL import Image
import tifffile as tiff
import math

paths = [ ]
tk = []
for k in range(1,17):
	paths.append('exposure%d.nef' % k )
	tk.append(2**(k-12))
print(k)
print(paths)
print(tk)
i = 0;
########## section 3.a <processing raw images and resizing the images >#######################
for path in paths:
	i= i +1	
	with rawpy.imread(path) as raw:
   		 rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
	xnew,ynew=rgb.shape[1]/10,rgb.shape[0]/10
	xnew = int(xnew)
	ynew = int(ynew)
	#print (xnew)
	rgb=cv2.resize(rgb,(xnew,ynew))
	imageio.imsave('processed_exposure%d.tiff' % i,rgb)
####################################

########## section 3.b <calculating hdr image using the formula >#######################

Ihdr=np.zeros((400,600,3))
Ihdr0=0
Ihdr1=0


for i in range(0,400):
	for j in range(0,600):
		for c in range(0,3):
			for k in range(1,17):
				Pex=cv2.imread('processed_exposure'+str(k)+'.tiff')
				#print(Pex.dtype)
				#print(Pex[i,j][c])
				norm=Pex[i,j][c]/(255.0)
				#print(norm)
				Ihdr0=Ihdr0+math.exp((-4*(norm-0.5)**2)/(0.5**2))*((Pex[i,j][c])/((1.0/2048)*(2**(k-1))))
				#print("Hello")
				#print(Ihdr0)
				Ihdr1=Ihdr1+math.exp((-4*(norm-0.5)**2)/(0.5**2))
				#print(Ihdr1)
			Ihdr[i,j][c]=Ihdr0/Ihdr1
			Ihdr0=0
			Ihdr1=0
####################################

########## section 3.c <displaying and saving the tonemapped image >#######################
cv2.imshow('image',lhdr)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('HDR_phototonemap.png',Ihdr)
####################################

########## section 3. d<built In tonemapping  >#######################
paths = [ ]
tk = []
ps =[]
for k in range(1,17):
	paths.append('exposure%d.nef' % k )
	tk.append(2**(k-12))
i = 0;
for path in paths:
	i= i +1	
	with rawpy.imread(path) as raw:
   		 rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
	xnew,ynew=rgb.shape[1]/10,rgb.shape[0]/10
	xnew = int(xnew)
	ynew = int(ynew)
	#print (xnew)
	rgb=cv2.resize(rgb,(xnew,ynew))
	imageio.imsave('processed_exposure%d.tiff' % i,rgb)

for k in range(1,17):
	ps.append('processed_exposure%d.tiff' % k )
img_list = [cv2.imread(fn) for fn in ps]
ks = np.array(tk)
ks = np.array(tk)
print(type(ks))
alignMTB = cv2.createAlignMTB()
alignMTB.process(img_list, img_list)
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=ks)


#Tonemap HDR image
tonemap2 = cv2.createTonemapDrago(1.2,0.7)
res_robertson = tonemap2.process(hdr_robertson.copy())

 # Convert datatype to 8-bit and save
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
cv2.imwrite("ldr_robertson_BUILTIN.jpg", res_robertson_8bit)
#cv2.imwrite("ldr_robertsontry.jpg", res_robertson)
###############################################