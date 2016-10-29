import timeit
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

test = cv2.imread('Train/img_2.jpg',0)
rows, cols = test.shape

x = 7								#Window size/2
red = 16							#PCA Reduce components
temp = rows*cols					#No. of pixels to test

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
	kp = cv2.KeyPoint(crow, ccol, 20)
	_, des1 = surf.compute(img, [kp])
	_, des2 = surf.compute(octave2, [kp])
	_, des3 = surf.compute(octave3, [kp])
	if len(_)>0:
		ind_ftr[4*x*x:4*x*x+128] = des1.mean(0)
		ind_ftr[4*x*x+128:4*x*x+256] = des2.mean(0)
		ind_ftr[4*x*x+256:4*x*x+384] = des3.mean(0)
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
rows, cols = test.shape
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0,nOctaves=3,extended=True)
octave2 = cv2.GaussianBlur(test, (0, 0), 1)
octave3 = cv2.GaussianBlur(test, (0, 0), 2)
features,pixels,N=[],[],[]
for crow in range(0,rows+1):
	for ccol in range(0,cols+1):
		pixels,features = ftr_ext(crow,ccol,octave2,octave3,test)
#PCA
pca_ftr_test = pca_(features,red)
stop = timeit.default_timer()
print ("Test - Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",pca_ftr_test.shape)

dft = pd.DataFrame(np.concatenate((pixels, pca_ftr_test), 1))
dft.to_csv('ftr_ext_test.csv', sep=',',header=False,index=False)
