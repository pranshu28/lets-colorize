import scipy.stats as stats
import timeit
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

labeled = np.matrix(pd.read_csv('labeled.csv', sep=',',header=None))
gray = cv2.imread('Train/img.jpg',0)
rows, cols = gray.shape

x = 5								#Window size/2
red = 32							#PCA Reduce components
perc = 10							#Percentage of random pixels to train

def ftr_ext(crow,ccol,octave1,octave2,img):
	ind_ftr=[None] * (4*x*x+(128*3)+4)
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

	#FFT
	ft=np.abs(np.fft.fft(mask.flatten()))
	if 4*x*x!=len(ft):
		flag = 0 
	ind_ftr[:4*x*x] = ft

	#SURF
	kp = cv2.KeyPoint(crow, ccol, 2*x)
	_, des1 = surf.compute(img, [kp])
	_, des2 = surf.compute(octave2, [kp])
	_, des3 = surf.compute(octave3, [kp])
	if len(_)>0:
		ind_ftr[4*x*x:4*x*x+128] = des1[0]
		ind_ftr[4*x*x+128:4*x*x+256] = des2[0]
		ind_ftr[4*x*x+256:4*x*x+384] = des3[0]
	else: 
		flag = 0

	#Position
	ind_ftr[-4] = crow/rows
	ind_ftr[-3] = ccol/cols
	#Mean and Variance
	ind_ftr[-2] = np.mean(mask)
	ind_ftr[-1] = np.var(mask)
	
	#Collect
	ind_ftr = np.asarray(ind_ftr)
	if flag==1 and np.isfinite(ind_ftr.sum()):
		pixels.append([crow,ccol])
		features.append(np.asarray(ind_ftr))
		if len(pixels)%1000==0:
			print(len(pixels))#,crow,ccol,mask.shape,len(ind_ftr))
	return pixels,features

def pca_(features,r):
	features = np.asmatrix(features)
	pca = PCA(n_components=r)
	pca.fit(features)
	return pca.transform(features)

#Feature Extraction
start = timeit.default_timer()
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0,nOctaves=3,extended=True)
octave2 = cv2.GaussianBlur(gray, (0, 0), 1)
octave3 = cv2.GaussianBlur(gray, (0, 0), 2)
features,pixels,N=[],[],[]

lim = int(rows*cols*.01*perc)+1
sigma = 1000
crow = stats.truncnorm((0 - rows/2) / sigma, (rows - rows/2) / sigma, loc=rows/2, scale=sigma)
ccol = stats.truncnorm((0 - cols/2) / sigma, (cols - cols/2) / sigma, loc=cols/2, scale=sigma)
crows = [int(i) for i in crow.rvs(lim)]
ccols = [int(i) for i in ccol.rvs(lim)]
for i in range(lim):
	pixels,features = ftr_ext(crows[i],ccols[i],octave2,octave3,gray)

#PCA
pca_ftr = pca_(features,red)
stop = timeit.default_timer()
print ("Train - Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",pca_ftr.shape)

pixels = np.matrix(pixels).T
train_svc = labeled[pixels[0,:],pixels[1,:]].reshape(pca_ftr.shape[0],1)
df = pd.DataFrame(np.concatenate((train_svc, pca_ftr), 1))
df.to_csv('ftr_ext_train.csv', sep=',',header=False,index=False)