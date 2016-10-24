from random import randint
import timeit
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

x = 20								#Window size/2
red = 32							#PCA Reduce components
perc = 1.5							#Percentage of random pixels to train

def ftr_ext(crow,ccol,img,color=False):
	ind_ftr=[None] * (4*x*x+(128*3)+2)
	flag = 1

	#Window
	xl,xh,yl,yh=crow-x,crow+x,ccol-x,ccol+x
	if xl<0:
		xl=0
		xh=2*x
	elif xh>rows:
		xh=rows
		xl=rows-x
	if yl<0:
		yl=0
		yh=2*x
	elif yh>cols:
		yh=cols
		yl=cols-x
	mask=img[xl:xh, yl:yh]
	if color==True:
		mask = mask[:,:,0]

	#FFT
	dft = cv2.dft(np.float32(mask),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	fft = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
	ft = list(np.reshape(fft, (np.product(fft.shape),)))
	if 4*x*x!=len(ft):
		flag = 0 
	ind_ftr[:4*x*x] = ft

	#SURF
	octave2 = cv2.GaussianBlur(img, (0, 0), 1)
	octave3 = cv2.GaussianBlur(img, (0, 0), 2)
	surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0,nOctaves=3,extended=True)
	kp1, des1 = surf.detectAndCompute(mask,None,useProvidedKeypoints = False)
	kp2, des2 = surf.detectAndCompute(octave2,None,useProvidedKeypoints = False)
	kp3, des3 = surf.detectAndCompute(octave3,None,useProvidedKeypoints = False)
	if len(kp1)>0 and len(kp2)>0 and len(kp3)>0:
		ind_ftr[4*x*x:4*x*x+128] = des1.mean(0)
		ind_ftr[4*x*x+128:4*x*x+256] = des2.mean(0)
		ind_ftr[4*x*x+256:4*x*x+384] = des3.mean(0)
	else: 
		flag = 0

	#Mean and Variance
	ind_ftr[-2] = np.mean(mask)
	ind_ftr[-1] = np.var(mask)
	
	#Collect
	ind_ftr = np.asarray(ind_ftr)
	if flag==1 and np.isfinite(ind_ftr.sum()):
		pixels.append([crow,ccol])
		features.append(np.asarray(ind_ftr))
		#print(crow,ccol,mask.shape,len(ind_ftr))
	return pixels,features

def pca_(features,r):
	features = np.asmatrix(features)
	pca = PCA(n_components=r)
	pca.fit(features)
	return pca.transform(features)



quant_train = cv2.imread('quant_train.jpg',1)

#Feature Extraction
start = timeit.default_timer()
rows, cols = quant_train.shape[:-1]
features,pixels=[],[]
i=0
while len(pixels)<int(rows*cols*.01*perc)+1:
	crow,ccol = randint(0,rows+1),randint(0,cols+1)
	if [crow,ccol] not in pixels:
		print(i)
		i=i+1
		pixels,features = ftr_ext(crow,ccol,quant_train,color=True)

#PCA
pca_ftr = pca_(features,red)
stop = timeit.default_timer()
print ("Train - Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",pca_ftr.shape)

pixels = np.matrix(pixels).T
train_svc = quant_train[pixels[0,:],pixels[1,:],:].reshape(pca_ftr.shape[0],3)
df = pd.DataFrame(np.concatenate((train_svc, pca_ftr), 1))
df.to_csv('ftr_ext_train.csv', sep=',',header=False,index=False)
