import timeit
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq
import cv2

class col_quant(object):
	def __init__(self, train='Train/img.jpg', ncolors=16):
		self.k=ncolors
		self.train=cv2.imread(train,1)

	def color_quant(self,img,k):
		img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
		a,b = np.float32(img[:,:,1]),np.float32(img[:,:,2])
		pixel = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
		centroids,_ = kmeans(pixel,self.k)
		qnt,_ = vq(pixel,centroids)
		color_to_label_map = {c:i for i,c in enumerate([tuple(i) for i in centroids])} 
		label_to_color_map = dict(zip(color_to_label_map.values(),color_to_label_map.keys()))
		return qnt,label_to_color_map

	#Color Quantization
	def run(self):
		start = timeit.default_timer()
		qnt,label = self.color_quant(self.train,self.k)
		self.labeled = pd.DataFrame(qnt.reshape(self.train.shape[:-1]))
		self.colors = pd.DataFrame(label).transpose()
		stop = timeit.default_timer()
		print ("Train -	Color Quantization: Done in ",stop-start," sec")

	#Save
	def export(self):
		self.colors.to_csv('colors.csv', sep=',',header=False,index=False)
		self.labeled.to_csv('labeled.csv', sep=',',header=False,index=False)