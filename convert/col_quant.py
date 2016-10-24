import timeit
import numpy as np
import cv2
import pandas as pd

k = 16								#K-means no. of discrete colors

def color_quant(img,k):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
	Z = img[:,:,1:].reshape((-1,2))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,color_list=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	color_list = np.uint8(color_list)
	res = color_list[label.flatten()]
	return pd.DataFrame(color_list),np.concatenate((img[:,:,0:-2], res.reshape((img[:,:,1:].shape))), 2)

train = cv2.imread('img_7.jpg',1)

#Color Quantization
start = timeit.default_timer()
colors,quant_train = color_quant(train,k)
stop = timeit.default_timer()
print ("Train - Color Quantization: Done in ",stop-start," sec")

cv2.imwrite('quant_train.jpg',quant_train)
