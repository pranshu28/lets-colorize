import timeit
import numpy as np
import pandas as pd

import scipy.stats as stats
from sklearn.decomposition import PCA
import cv2

import warnings
warnings.filterwarnings("ignore")

class train_ftr_ext(object):

	def __init__(self, train_ftr, x=5,red=32,perc=10):
		self.labeled = np.matrix(pd.read_csv('labeled.csv', sep=',',header=None))
		self.train_ftr = cv2.imread(train_ftr,0)
		self.rows, self.cols = self.train_ftr.shape
		self.x = x
		self.red=red
		self.perc=perc
		self.features,self.pixels,self.N=[],[],[]

	def ftr_ext(self,crow,ccol,octave1,octave2,img):
		ind_ftr=[None] * (4*self.x*self.x+(128*3)+4)
		flag = 1

		#Window
		xl,xh,yl,yh=crow-self.x,crow+self.x,ccol-self.x,ccol+self.x
		if xl<0:
			xl=0
			xh=2*self.x
		elif xh>self.rows:
			xh=self.rows
			xl=self.rows-self.x
		if yl<0:
			yl=0
			yh=2*self.x
		elif yh>self.cols:
			yh=self.cols
			yl=self.cols-self.x
		mask=img[xl:xh, yl:yh]

		#FFT
		ft=np.abs(np.fft.fft(mask.flatten()))
		if 4*self.x*self.x!=len(ft):
			flag = 0 
		ind_ftr[:4*self.x*self.x] = ft

		#SURF
		kp = cv2.KeyPoint(crow, ccol, 2*self.x)
		surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0,nOctaves=3,extended=True)
		_, des1 = surf.compute(img, [kp])
		_, des2 = surf.compute(octave1, [kp])
		_, des3 = surf.compute(octave2, [kp])
		if len(_)>0:
			ind_ftr[4*self.x*self.x:4*self.x*self.x+128] = des1[0]
			ind_ftr[4*self.x*self.x+128:4*self.x*self.x+256] = des2[0]
			ind_ftr[4*self.x*self.x+256:4*self.x*self.x+384] = des3[0]
		else: 
			flag = 0

		#Position
		ind_ftr[-4] = crow/self.rows
		ind_ftr[-3] = ccol/self.cols
		#Mean and Variance
		ind_ftr[-2] = np.mean(mask)
		ind_ftr[-1] = np.var(mask)
		
		#Collect
		ind_ftr = np.asarray(ind_ftr)
		if flag==1 and np.isfinite(ind_ftr.sum()):
			self.pixels.append([crow,ccol])
			self.features.append(np.asarray(ind_ftr))
			#if len(self.pixels)%1000==0:
			#	print(len(self.pixels))#,crow,ccol,mask.shape,len(ind_ftr))
		return self.pixels,self.features

	def pca_(self,features,r):
		self.features = np.asmatrix(self.features)
		pca = PCA(n_components=r)
		pca.fit(self.features)
		return pca.transform(self.features)

	#Feature Extraction
	def run(self):
		start = timeit.default_timer()
		octave2 = cv2.GaussianBlur(self.train_ftr, (0, 0), 1)
		octave3 = cv2.GaussianBlur(self.train_ftr, (0, 0), 2)
		lim = int(self.rows*self.cols*.01*self.perc)+1
		sigma = 1000
		crow = stats.truncnorm((0 - self.rows/2) / sigma, (self.rows - self.rows/2) / sigma, loc=self.rows/2, scale=sigma)
		ccol = stats.truncnorm((0 - self.cols/2) / sigma, (self.cols - self.cols/2) / sigma, loc=self.cols/2, scale=sigma)
		crows = [int(i) for i in crow.rvs(lim)]
		ccols = [int(i) for i in ccol.rvs(lim)]
		for i in range(lim):
			self.pixels,self.features = self.ftr_ext(crows[i],ccols[i],octave2,octave3,self.train_ftr)

		#PCA
		self.pca_ftr = self.pca_(self.features,self.red)
		stop = timeit.default_timer()
		print ("Train -	Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",self.pca_ftr.shape)
		return stop-start

	#Save
	def export(self):
		self.pixels = np.matrix(self.pixels).T
		train_svc = self.labeled[self.pixels[0,:],self.pixels[1,:]].reshape(self.pca_ftr.shape[0],1)
		df = pd.DataFrame(np.concatenate((train_svc, self.pca_ftr), 1))
		df.to_csv('ftr_ext_train.csv', sep=',',header=False,index=False)