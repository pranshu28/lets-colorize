import timeit
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings("ignore")

reg = 1.0							#Regularization in SVMs

#Classification
df = pd.read_csv('ftr_ext_train.csv', sep=',',header=None)
train_svc = df.ix[:,0].as_matrix()
pca_ftr = df.ix[:,1:].as_matrix()
start = timeit.default_timer()
Y = MultiLabelBinarizer().fit_transform(train_svc.reshape(len(train_svc),1)).T
clfs=[]
for i,y in enumerate(Y):
	clf = LinearSVC(C=reg)
	print(i)
	clf.fit(pca_ftr,y)
	clfs.append(clf)
stop = timeit.default_timer()
print ("Train - Classification: Done in ",stop-start," sec - Feature: ",pca_ftr.shape,"	Data: ",Y.shape)

#Predict
df = pd.read_csv('ftr_ext_test.csv', sep=',',header=None)
pixels = df.ix[:,:1].as_matrix()
pca_ftr_test = df.ix[:,2:].as_matrix()
start = timeit.default_timer()
costs=[]
for i,x in enumerate(Y):
	costs.append(clfs[i].decision_function(pca_ftr_test))
stop = timeit.default_timer()
print ("Test - Prediction: Done in ",stop-start," sec - Test: ",pca_ftr_test.shape,"	Result: ",np.matrix(costs).shape)

dfpr = pd.DataFrame(np.concatenate((pixels, np.matrix(costs).T), 1))
dfpr.to_csv('pred_cost.csv', sep=',',header=False,index=False)
