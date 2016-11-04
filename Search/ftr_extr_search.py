import cv2
import numpy as np
# import ufp.image

def main(img_src):
	#Fuction to get Luminance Histogram
	def lum_hist(img,bins):
		return cv2.calcHist([img], [0], None, [bins], [0, bins]).flatten()

	#Fuction to get Luminance subsampling
	def lum_sub(img,x,y):
		return cv2.resize(img[:,:,0],(x,y)).flatten()

	# Function to quantize image
	# def quant(img,k):
	# 	return (img*float(k-1))/255

	img = cv2.imread(img_src,1)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

	#QUANTIZTION goes here(Colorspace reuction to 128 gray level)
	#img_q = quant(img,128).astype('int32')


	#High-pass Filter application
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	abs_lap = np.absolute(laplacian)
	lap_uint8 = np.uint8(abs_lap)
	#lap_q = quant(lap_uint8,128)
	# print np.max(img_q)

	#sobel = cv2.Sobel(img_q, 0, 1, 1)

	#print np.max(lap_q)
	# print img.shape,img_q.shape

	#Collecting feature vectors of size 128
	f1 = lum_hist(img,128) #Feature vector consisting of Luminosity histogram
	f2 = lum_sub(img,8,16) #Feature vector consisting of Luminosity subsampling	
	f3 = np.concatenate((lum_hist(img,64),(lum_hist(lap_uint8,64)))) #Feature vector consisting of Luminosity+gradient histogram 
	f4 = np.concatenate((lum_sub(img,8,8),(lum_sub(lap_uint8,8,8)))) #Feature vector consisting of Luminosity+gradient subsampling

	# print "f1----------"
	# print f1
	# print "f2----------"
	# print f2
	# print "f3----------"
	# print f3
	# print "f4----------"
	# print f4
	# # #scv2.imshow("test",sobel)
	# # cv2.waitKey(0)
	# # cv2.destroyAllWindows()
	return f1,f2,f3,f4	

if __name__=="__main__":
	main()