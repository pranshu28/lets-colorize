import numpy as np
import pandas as pd
import cv2
import pygco

import warnings
warnings.filterwarnings("ignore")

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
			c1 = np.array(colors[ii,1:])
			c2 = np.array(colors[jj,1:])
			pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)
	label_costs_int32 = np.ascontiguousarray(label_costs).astype('int32')
	pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
	vv_int32 = edges.astype('int32')
	vh_int32 = edges.astype('int32')
	new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap') 
	return new_labels


test = cv2.imread('img_8.jpg',0)
df = pd.read_csv('pred.csv', sep=',')
dfpr = pd.read_csv('pred_prob.csv', sep=',')
colors = pd.read_csv('colors.csv', sep=',',header=None).as_matrix()
pixels = df.ix[:,:2].as_matrix()
pred = df.ix[:,2:].as_matrix()
pred_prob = dfpr.ix[:,2:].as_matrix()

rows, cols = test.shape
label_costs = np.zeros((rows,cols,len(colors)))
for i,x in enumerate(pixels):
	cost = -100*pred_prob[i]
	label_costs[x[0],x[1]] = np.array(cost).astype(int)
edges = get_edges(test)
output_labels = graphcut(test,edges,label_costs, l=1)

#Colorization
output_a = np.zeros((rows,cols))
output_b = np.zeros((rows,cols))
for i in pixels[:,0]:
	for j in pixels[:,1]:
		a,b = colors[output_labels[i,j],1:]
		output_a[i,j] = a
		output_b[i,j] = b

output_img = cv2.cvtColor(cv2.merge((test, np.uint8(output_a), np.uint8(output_b))), cv2.COLOR_Lab2RGB)

cv2.imshow('Test',output_img)
cv2.imwrite('RESULT.jpg',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
