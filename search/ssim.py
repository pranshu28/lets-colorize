import cv2
import os
from skimage.measure import structural_similarity as ssim
#from image_match.goldberg import ImageSignature
import itertools

#gis = ImageSignature()
#main_img = gis.generate_signature('/home/deathstroke/projects/G2RGB/testergrey/fruit-still-life-1.jpg')
main_img = cv2.imread('/home/deathstroke/projects/G2RGB/testergrey/fruit-still-life-1.jpg')
main_img = cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
dim = (200,200)
main_img = cv2.resize(main_img,dim)
os.chdir('/home/deathstroke/projects/G2RGB/')
images = os.listdir('testimages')
score=[]
for each in images:
	comp_img = cv2.imread('/home/deathstroke/projects/G2RGB/testimages/%s' %(each))
	comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
	comp_img = cv2.resize(comp_img,dim)
	b = ssim(main_img,comp_img)
	score.append(b)
# for i,j in itertools.izip(name,score):
#     print i,' : ',j
max_score = max(score)
ind = score.index(max_score)
nameoffile = images[ind]
print nameoffile,' : ',max_score