import timeit
import numpy as np
import pandas as pd
import pygco
import cv2

class colorize(object):

	def __init__(self, original = 'Train/img_2.jpg'):
		self.test = cv2.imread(original,0)
		self.original = cv2.imread(original,1)
		self.dfpr = pd.read_csv('pred_cost.csv', sep=',',header=None)
		self.colors = pd.read_csv('colors.csv', sep=',',header=None).as_matrix()
		self.pixels = self.dfpr.ix[:,:2].as_matrix()
		self.pred_cost = self.dfpr.ix[:,2:].as_matrix()
		self.rows, self.cols = self.test.shape
	
	def get_edges(self,blur_width=3):
		img_blurred = cv2.GaussianBlur(self.test, (0, 0), blur_width)
		vh = cv2.Sobel(img_blurred, -1, 1, 0)
		vv = cv2.Sobel(img_blurred, -1, 0, 1)
		return 0.5*vv + 0.5*vh

	def graphcut(self,edges,label_costs, l=100):
		num_classes = len(self.colors)
		pairwise_costs = np.zeros((num_classes, num_classes))
		for ii in range(num_classes):
			for jj in range(num_classes):
				c1 = np.array(self.colors[ii])
				c2 = np.array(self.colors[jj])
				pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)
		label_costs_int32 = np.ascontiguousarray(label_costs).astype('int32')
		pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
		vv_int32 = edges.astype('int32')
		vh_int32 = edges.astype('int32')
		new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap') 
		#new_labels = pygco.cut_simple(label_costs_int32, pairwise_costs_int32, n_iter=10, algorithm='swap') 
		return new_labels

	#Colorization
	def color(self):
		start = timeit.default_timer()
		label_costs = np.zeros((self.rows,self.cols,len(self.colors)))
		for i,x in enumerate(self.pixels):
			cost = -100*self.pred_cost[i]
			label_costs[x[0],x[1]] = np.array(cost).astype(int)
		edges = self.get_edges()
		output_labels = self.graphcut(edges,label_costs, l=1)
		pd.DataFrame(output_labels).to_csv('output_labeled.csv', sep=',',header=False,index=False)
		#y = np.bincount(output_labels.reshape(self.rows*self.cols))
		#ii = np.nonzero(y)[0]
		#print(np.vstack((ii,y[ii])).T)
		self.ab = self.colors[output_labels]
		self.output_img = cv2.cvtColor(cv2.merge((self.test, np.uint8(self.ab[:,:,0]), np.uint8(self.ab[:,:,1]))), cv2.COLOR_Lab2RGB)
		stop = timeit.default_timer()
		print ("Test - Colorization: Done in ",stop-start," sec - ",self.output_img.shape)
		return output_img

	#Compare
	def compare(self):
		diff = cv2.subtract(self.original, self.output_img)
		print("Error : ",np.std(diff))
		return diff

	#Save
	def export(self):
		cv2.imwrite('RESULT.jpg',self.output_img)
