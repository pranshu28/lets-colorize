import timeit
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

reg = 1.5							#Regularization in SVMs

#Classification
df = pd.read_csv('ftr_ext_train.csv', sep=',',header=None)
train_svc = df.ix[:,1:2].as_matrix()
colors = np.vstack({tuple(row) for row in train_svc})
labels=[]
for i,col in enumerate(train_svc):
	lab = np.where(np.all(colors==col,axis=1))
	labels.append(lab[0])
pca_ftr = df.ix[:,3:].as_matrix()
start = timeit.default_timer()
Y = MultiLabelBinarizer().fit_transform(labels)
clf = OneVsRestClassifier(SVC(C=reg,kernel='linear',probability=True))
clf.fit(pca_ftr, Y)
stop = timeit.default_timer()
print ("Train - Classification: Done in ",stop-start," sec - Feature: ",pca_ftr.shape,"	Data: ",Y.shape)

#Predict
df = pd.read_csv('ftr_ext_test.csv', sep=',',header=None)
pixels = df.ix[:,:1].as_matrix()
pca_ftr_test = df.ix[:,2:].as_matrix()
start = timeit.default_timer()
predict = np.matrix(clf.predict(pca_ftr_test))
predict_prob = np.matrix(clf.predict_proba(pca_ftr_test))
stop = timeit.default_timer()
print ("Test - Prediction: Done in ",stop-start," sec - Test: ",pca_ftr_test.shape,"	Result: ",predict.shape,predict_prob.shape)

df = pd.DataFrame(np.concatenate((pixels, predict), 1))
df.to_csv('pred.csv', sep=',',header=False,index=False)
pd.DataFrame(colors).to_csv('colors.csv', sep=',',header=None)

dfpr = pd.DataFrame(np.concatenate((pixels, predict_prob), 1))
dfpr.to_csv('pred_prob.csv', sep=',',header=False,index=False)
