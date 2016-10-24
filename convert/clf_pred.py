import timeit
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

reg = 1.0							#Regularization in SVMs


#Classification
df = pd.read_csv('ftr_ext_train.csv', sep=',')
train_svc = df.ix[:,1:4].as_matrix()
pca_ftr = df.ix[:,4:].as_matrix()
start = timeit.default_timer()
Y = MultiLabelBinarizer().fit_transform(train_svc[:,1:])
clf = OneVsRestClassifier(SVC(C=reg,kernel='linear',probability=True))
clf.fit(pca_ftr, Y)
stop = timeit.default_timer()
print ("Train - Classification: Done in ",stop-start," sec - Feature: ",pca_ftr.shape,"	Data: ",Y.shape)


#Predict
df = pd.read_csv('ftr_ext_test.csv', sep=',')
pixels = df.ix[:,1:3].as_matrix()
pca_ftr_test = df.ix[:,3:].as_matrix()
start = timeit.default_timer()
predict = np.matrix(clf.predict(pca_ftr_test))
predict_prob = np.matrix(clf.predict_proba(pca_ftr_test))
stop = timeit.default_timer()
print ("Test - Prediction: Done in ",stop-start," sec - Test: ",pca_ftr_test.shape,"	Result: ",predict.shape,predict_prob.shape)

df = pd.DataFrame(np.concatenate((pixels, predict), 1))
df.to_csv('pred.csv', sep=',')

dfpr = pd.DataFrame(np.concatenate((pixels, predict_prob), 1))
dfpr.to_csv('pred_prob.csv', sep=',')
