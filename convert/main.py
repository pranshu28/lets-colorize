from col_quant import col_quant
from train_ftr_ext import train_ftr_ext
from test_ftr_ext import test_ftr_ext
from clf_pred import clf_pred
from colorize import colorize

if __name__ == '__main__':

	def show(img):
		cv2.imshow('Image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 

	train = 'Train/img.jpg'
	test='Train/img_2.jpg'
	k = 2								#No. of colors
	x = 5								#Window size/2
	red = 32							#PCA Reduce components
	perc = 10							#Percentage of random pixels to train
	reg=1.0								#Regularization in Classification

	quant = col_quant(train=train, ncolors=k)
	quant.run()
	quant.export()
	train_ftr = train_ftr_ext(train_ftr=train,x=x,red=red,perc=perc)
	train_ftr.run()
	train_ftr.export()
	test_ftr = test_ftr_ext(test_ftr=test,x=x,red=red)
	test_ftr.run()
	test_ftr.export()
	svm = clf_pred(reg=reg)
	svm.clf()
	svm.pred()
	svm.export()
	colorizer = colorize(original=test)
	output = colorizer.color()
	error = colorizer.compare()
	colorizer.export()
	show(train)
	show(test)
	show(output)
	show(error)