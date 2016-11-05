import timeit
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import cv2

class clf_pred(object):

	def __init__(self,reg=1.0):
		self.reg=reg
		self.clfs=[]
		self.costs=[]

	#Classification
	def clf(self):
		df = pd.read_csv('ftr_ext_train.csv', sep=',',header=None)
		train_svc = df.ix[:,0].as_matrix()
		pca_ftr = df.ix[:,1:].as_matrix()
		start = timeit.default_timer()
		self.Y = MultiLabelBinarizer().fit_transform(train_svc.reshape(len(train_svc),1)).T
		for i,y in enumerate(self.Y):
			clf = LinearSVC(C=self.reg)
			print(i)
			clf.fit(pca_ftr,y)
			self.clfs.append(clf)
		stop = timeit.default_timer()
		print ("Train - Classification: Done in ",stop-start," sec - Feature: ",pca_ftr.shape,"	Data: ",self.Y.shape)

	#Predict
	def pred(self):
		df = pd.read_csv('ftr_ext_test.csv', sep=',',header=None)
		self.pixels = df.ix[:,:1].as_matrix()
		pca_ftr_test = df.ix[:,2:].as_matrix()
		start = timeit.default_timer()
		for i,x in enumerate(self.Y):
			self.costs.append(self.clfs[i].decision_function(pca_ftr_test))
		stop = timeit.default_timer()
		print ("Test - Prediction: Done in ",stop-start," sec - Test: ",pca_ftr_test.shape,"	Result: ",np.matrix(self.costs).shape)

	#Save
	def export(self):
		dfpr = pd.DataFrame(np.concatenate((self.pixels, np.matrix(self.costs).T), 1))
		dfpr.to_csv('pred_cost.csv', sep=',',header=False,index=False)
