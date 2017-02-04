from col_quant import col_quant
from train_ftr_ext import train_ftr_ext
from test_ftr_ext import test_ftr_ext
from clf_pred import clf_pred
from colorize import colorize
import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':

	train = 'Train/img_6.jpg'
	test='Train/img_6.jpg'
	k = 16								#No. of colors
	x = 10								#Window size/2
	red = 32							#PCA Reduce components
	perc = 10							#Percentage of random pixels to train
	reg = 1.0							#Regularization in Classification
	sobel = True						#Use Sobel weights

	quant = col_quant(train=train, ncolors=k)
	t1 = quant.run()
	quant.export()

	train_ftr = train_ftr_ext(train_ftr=train,x=x,red=red,perc=perc)
	t2 = train_ftr.run()
	train_ftr.export()

	t3 = ""
	test_ftr = test_ftr_ext(test_ftr=test,x=x,red=red)
	t3 = test_ftr.run()
	test_ftr.export()

	svm = clf_pred(reg=reg)
	t4 = svm.clf()
	t5 = svm.pred()
	svm.export()

	colorizer = colorize(original=test,sobel=sobel)
	t6,output = colorizer.color()
	error = colorizer.compare()
	colorizer.export()

	pd.DataFrame([[train,test,k,x,red,perc,reg,sobel,t1,t2,t3,t4,t5,t6,np.std(error)]]).to_csv("Record.csv",mode='a',index=False,header=False)

	cv2.imshow('Output',output)
	cv2.imshow('Error',error)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
