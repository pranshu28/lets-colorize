import numpy as np
import pandas as pd
import cv2
from gco_python import pygco

def get_edges(img, blur_width=3):
	img_blurred = cv2.GaussianBlur(img, (0, 0), blur_width)
	vh = cv2.Sobel(img_blurred, -1, 1, 0)
	vv = cv2.Sobel(img_blurred, -1, 0, 1)
	return 0.5*vv + 0.5*vh

def graphcut(img,edges,label_costs, l=100):
	num_classes = len(colors)
    #calculate pariwise potiential costs (distance between color classes)
	pairwise_costs = np.zeros((num_classes, num_classes))
	for ii in range(num_classes):
		for jj in range(num_classes):
			c1 = np.array(colors[1:,ii])
			c2 = np.array(colors[1:,jj])
			pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)
    
	label_costs_int32 = (100*label_costs).astype('int32')
	pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
	vv_int32 = edges.astype('int32')
	vh_int32 = edges.astype('int32')
	new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap') 
	return new_labels


test = cv2.imread('img_8.jpg',0)
df = pd.read_csv('pred.csv', sep=',')
dfpr = pd.read_csv('pred_prob.csv', sep=',')
colors = pd.read_csv('colors.csv', sep=',',header=None).as_matrix()
pixels = df.ix[:,1:3].as_matrix()
pred = df.ix[:,3:].as_matrix()
pred_prob = dfpr.ix[:,3:].as_matrix()
edges = get_edges(img)
output_labels = graphcut(test,edges,pred_prob, l=1)

#Colorization
for i in pixels[:,0]:
	for j in pixels[:,1]:
		a,b = colors[output_labels[i,j],1:]
		output_a[i,j] = a
		output_b[i,j] = b

output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv2.CV_Lab2RGB)

cv2.imshow('Test',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()