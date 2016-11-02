import timeit
import numpy as np
import pandas as pd
import cv2
import pygco
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

test = cv2.imread('Train/img_2.jpg',0)
original = cv2.imread('Train/img_2.jpg',1)
dfpr = pd.read_csv('pred_cost.csv', sep=',',header=None)
colors = pd.read_csv('colors.csv', sep=',',header=None).as_matrix()

def get_edges(img, blur_width=3):
	img_blurred = cv2.GaussianBlur(img, (0, 0), blur_width)
	vh = cv2.Sobel(img_blurred, -1, 1, 0)
	vv = cv2.Sobel(img_blurred, -1, 0, 1)
	return 0.5*vv + 0.5*vh

def graphcut(img,edges,label_costs, l=100):
	num_classes = len(colors)
	pairwise_costs = np.zeros((num_classes, num_classes))
	for ii in range(num_classes):
		for jj in range(num_classes):
			c1 = np.array(colors[ii])
			c2 = np.array(colors[jj])
			pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)
	label_costs_int32 = np.ascontiguousarray(label_costs).astype('int32')
	pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
	vv_int32 = edges.astype('int32')
	vh_int32 = edges.astype('int32')
	new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap') 
	#new_labels = pygco.cut_simple(label_costs_int32, pairwise_costs_int32, n_iter=10, algorithm='swap') 
	return new_labels

pixels = dfpr.ix[:,:2].as_matrix()
pred_cost = dfpr.ix[:,2:].as_matrix()
rows, cols = test.shape

#Colorization
start = timeit.default_timer()
label_costs = np.zeros((rows,cols,len(colors)))
for i,x in enumerate(pixels):
	cost = -100*pred_cost[i]
	label_costs[x[0],x[1]] = np.array(cost).astype(int)
edges = get_edges(test)
output_labels = graphcut(test,edges,label_costs, l=1)
pd.DataFrame(output_labels).to_csv('output_labeled.csv', sep=',',header=False,index=False)

y = np.bincount(output_labels.reshape(rows*cols))
ii = np.nonzero(y)[0]
print(np.vstack((ii,y[ii])).T)

ab = colors[output_labels]
output_img = cv2.cvtColor(cv2.merge((test, np.uint8(ab[:,:,0]), np.uint8(ab[:,:,1]))), cv2.COLOR_Lab2RGB)
stop = timeit.default_timer()
print ("Test - Colorization: Done in ",stop-start," sec - ",output_img.shape)

#Compare
ldiff = cv2.subtract(cv2.cvtColor(original, cv2.COLOR_RGB2Lab),cv2.merge((test, np.uint8(ab[:,:,0]), np.uint8(ab[:,:,1]))))
diff = cv2.subtract(original, output_img)
print("Error : ",np.std(diff))

cv2.imwrite('RESULT.jpg',output_img)
cv2.imshow('result',diff)
cv2.waitKey(0)
cv2.destroyAllWindows()