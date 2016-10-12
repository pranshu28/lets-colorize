#Ignore this file. Just kept to keep track of how I progressed

# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 	
 	return m,s
	# # setup the figure
	# fig = plt.figure(title)
	# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# # show first image
	# ax = fig.add_subplot(1, 2, 1)
	# plt.imshow(imageA, cmap = plt.cm.gray)
	# plt.axis("off")
 
	# # show the second image
	# ax = fig.add_subplot(1, 2, 2)
	# plt.imshow(imageB, cmap = plt.cm.gray)
	# plt.axis("off")
 
	# # show the images
	# plt.show()

main_img = cv2.imread('/home/deathstroke/Pictures/yosegrey.png')
main_img = cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
os.chdir('/home/deathstroke/Pictures/')
images = os.listdir('Wallpapers') 
print main_img.shape
dim = (1920,1080)
main_img = cv2.resize(main_img,dim)
print main_img.shape

names = []
mses = []
ssims = []

for each in images:
	comp_img = cv2.imread('/home/deathstroke/Pictures/Wallpapers/%s' %(each))
	comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
	comp_img = cv2.resize(comp_img,dim)
	m1,s1 = compare_images(main_img,comp_img)
	names.append(each)
	mses.append(m1)
	ssims.append(s1)

# for i,name in enumerate(names):
# 	print name,mses[i],ssims[i]
# max_ssim = max(ssims)
# max_ind = ssims.index(max_ssim)
# print names[max_ind]

min_mse = min(mses)
min_ind = mses.index(min_mse)
yose_ind = names.index('yosemite_mountains-2560x1440.jpg')
print names[min_ind], min_mse, names[yose_ind], mses[yose_ind]
