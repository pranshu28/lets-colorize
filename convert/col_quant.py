import timeit
import numpy as np
from scipy.cluster.vq import kmeans,vq
import cv2
import pandas as pd

train = cv2.imread('Train/img.jpg',1)

k = 16								#K-means no. of discrete colors

def color_quant(img,k):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
	a,b = np.float32(img[:,:,1]),np.float32(img[:,:,2])
	pixel = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
	centroids,_ = kmeans(pixel,k)
	qnt,_ = vq(pixel,centroids)
	color_to_label_map = {c:i for i,c in enumerate([tuple(i) for i in centroids])} 
	label_to_color_map = dict(zip(color_to_label_map.values(),color_to_label_map.keys()))
	return qnt,label_to_color_map

#Color Quantization
start = timeit.default_timer()
qnt,label = color_quant(train,k)
labeled = pd.DataFrame(qnt.reshape(train.shape[:-1]))
colors = pd.DataFrame(label).transpose()
stop = timeit.default_timer()
print ("Train - Color Quantization: Done in ",stop-start," sec")


colors.to_csv('colors.csv', sep=',',header=False,index=False)
labeled.to_csv('labeled.csv', sep=',',header=False,index=False)