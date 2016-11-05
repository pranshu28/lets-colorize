import timeit
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import cv2
import warnings
warnings.filterwarnings("ignore")

class test_ftr_ext(object):

	def __init__(self, test_ftr,x=5,red=32):
		self.labeled = np.matrix(pd.read_csv('labeled.csv', sep=',',header=None))
		self.test_ftr = cv2.imread(test_ftr,0)
		self.rows, self.cols = self.test_ftr.shape
		self.x = x
		self.red=red
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
		octave2 = cv2.GaussianBlur(self.test_ftr, (0, 0), 1)
		octave3 = cv2.GaussianBlur(self.test_ftr, (0, 0), 2)
		for crow in range(0,self.rows+1):
			for ccol in range(0,self.cols+1):
				self.pixels,self.features = self.ftr_ext(crow,ccol,octave2,octave3,self.test_ftr)
		#PCA
		self.pca_ftr_test_ftr = self.pca_(self.features,self.red)
		stop = timeit.default_timer()
		print ("Test -	Feature Extraction and PCA: Done in ",stop-start," sec - Reduced components: ",self.pca_ftr_test_ftr.shape)

	#Save
	def export(self):
		dft = pd.DataFrame(np.concatenate((self.pixels, self.pca_ftr_test_ftr), 1))
		dft.to_csv('ftr_ext_test.csv', sep=',',header=False,index=False)