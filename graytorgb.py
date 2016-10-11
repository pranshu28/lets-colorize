import timeit
from random import randint

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from skimage.filters import sobel

import warnings
warnings.filterwarnings("ignore")

k = 2								#K-means no. of discrete colors
red = 16							#PCA Reduce components
reg = 1.0							#Regularization in SVMs
x = 20								#Window size/2
perc = .015							#Percentage of random pixels to train
temp = 1000							#No. of pixels to test

def color_quant(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
	Z = img[:,:,1:].reshape((-1,2))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,color_list=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	color_list = np.uint8(color_list)
	res = color_list[label.flatten()]
	return np.concatenate((img[:,:,0:-2], res.reshape((img[:,:,1:].shape))), 2)

def ftr_ext(crow,ccol,img,color=False,test=False):
	ind_ftr=[None] * (4*x*x+128+2)
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
	surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0,nOctaves=3,extended=True)
	kp1, des1 = surf.detectAndCompute(mask,None,useProvidedKeypoints = False)
	#kp2, des2 = surf.detectAndCompute(mask[:,:,1],None,useProvidedKeypoints = False)
	#kp3, des3 = surf.detectAndCompute(mask[:,:,2],None,useProvidedKeypoints = False)
	if len(kp1)>0:# and len(kp2)>0 and len(kp3)>0:
		ind_ftr[4*x*x:4*x*x+128] = des1.mean(0)
		#ind_ftr[4*x*x+128:4*x*x+256] = des2.mean(0)
		#ind_ftr[4*x*x+256:4*x*x+384] = des3.mean(0)
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
	if test==True:
		return pixels,N.append([[crow-1,ccol-1],[crow-1,ccol],[crow-1,ccol+1],[crow,ccol-1],[crow,ccol+1],[crow+1,ccol-1],[crow+1,ccol],[crow+1,ccol+1]]),features
	return pixels,features

def pca_(features,r):
	features = np.asmatrix(features)
	pca = PCA(n_components=r)																		
	pca.fit(features)
	return pca.transform(features)


#--------------input images---------
tr=str(7)
t=str(8)
train = cv2.imread('img_'+tr+'.jpg',1)
test = cv2.imread('img_'+t+'.jpg',0)
print("Input Image: Done")


#--------------display-------------
#cv2.imshow('Train',train)
#cv2.imshow('Test',test)
print("Display Image: Done")


#--------------Train-------------
print("Train - ")
traing = cv2.cvtColor(train, cv2.COLOR_RGB2GRAY)
edge_sobel = sobel(traing)

#Color Quantization
start = timeit.default_timer()
quant_train = color_quant(train)
stop = timeit.default_timer()
print ("	Color Quantization: Done in ",stop-start," sec")


#Feature Extraction
start = timeit.default_timer()
rows, cols = train.shape[:-1]
features,pixels=[],[]
while len(pixels)<int(rows*cols*.01*perc)+1:																	
	crow,ccol = randint(0,rows+1),randint(0,cols+1)
	if [crow,ccol] not in pixels:
		pixels,features = ftr_ext(crow,ccol,train,color=True)
#PCA
pca_ftr = pca_(features,red)
stop = timeit.default_timer()
print ("	Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",pca_ftr.shape)


#Classification
start = timeit.default_timer()
pixels = np.matrix(pixels).T
train_svc = quant_train[pixels[0,:],pixels[1,:],:].reshape(pca_ftr.shape[0],3)																							
Y = MultiLabelBinarizer().fit_transform(train_svc[:,1:])
clf = OneVsRestClassifier(SVC(C=reg,kernel='linear',probability=True))
clf.fit(pca_ftr, Y)
stop = timeit.default_timer()
print ("	Classification: Done in ",stop-start," sec - Feature: ",pca_ftr.shape,"	Data: ",Y.shape)


#-------------Testing------------------
print("Test - ")

#Feature Extraction
start = timeit.default_timer()
rows, cols = test.shape
features,pixels,N=[],[],[]
i=0
for crow in range(0,rows+1):
	for ccol in range(0,cols+1):
		if i<temp:
			pixels,N,features = ftr_ext(crow,ccol,test,test=True)
			i+=1
#PCA
pca_ftr_test = pca_(features,red)
stop = timeit.default_timer()
print ("	Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",pca_ftr_test.shape)


#Predict
start = timeit.default_timer()
predict = clf.predict(pca_ftr_test)
stop = timeit.default_timer()
print ("	Prediction: Done in ",stop-start," sec - Test: ",pca_ftr_test.shape,"	Result: ",len(predict))


#Colorization

##MRF




#-------------Compare------------------
orig = cv2.imread('img_'+t+'.jpg',1)

#--------------Show the result----------
#cv2.imshow('final',final)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(features.shape,"\nFinal Display: Done")
#cv2.imwrite('final_'+tr+t,final)