import cv2
import os
from image_match.goldberg import ImageSignature
import itertools

gis = ImageSignature()
main_img = cv2.imread('/home/deathstroke/projects/G2RGB/testergrey/fruit-still-life-1.jpg')
main_img = cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
a = gis.generate_signature(main_img)
os.chdir('/home/deathstroke/projects/G2RGB/')
images = os.listdir('testimages')
score=[]
for each in images:
	comp_img = cv2.imread('/home/deathstroke/projects/G2RGB/testimages/%s' %(each))
	comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
	b = gis.generate_signature(comp_img)
	score.append(gis.normalized_distance(a, b))
# for i,j in itertools.izip(name,score):
#     print i,' : ',j
min_score = min(score)
ind = score.index(min_score)
nameoffile = images[ind]
print nameoffile,' : ',min_score