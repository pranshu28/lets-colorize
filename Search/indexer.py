import os
from os.path import join
import ftr_extr_search as ftr
import pandas as pd
import csv

directory="/media/deathstroke/Work/Study/Image-colorization/jpg/"

indexf1={}
indexf2={}
indexf3={}
indexf4={}
def find(directory):
	for (dirname,dirs,files) in os.walk(directory):
		for filename in files:
			if (filename.endswith(".jpg")):
				fullpath=join(dirname,filename)
				indexf1[fullpath],indexf2[fullpath],indexf3[fullpath],indexf4[fullpath]=ftr.main(fullpath)
	print "total number of photos in this directory %s"%len(indexf1)
	# return indexf1,indexf2,indexf3,indexf4

find(directory)

#Writing indexes into seperate csv files
with open('index1.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in indexf1.items():
       writer.writerow([key, value])

with open('index2.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in indexf2.items():
       writer.writerow([key, value])

with open('index3.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in indexf3.items():
       writer.writerow([key, value])

with open('index4.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in indexf4.items():
       writer.writerow([key, value])
# df1 = pd.DataFrame.from_dict(indexf1,orient='index')
# df1.to_csv("index1.csv")
# df2 = pd.DataFrame.from_dict(indexf2,orient='index')
# df2.to_csv("index2.csv")
# df3 = pd.DataFrame.from_dict(indexf3,orient='index')
# df3.to_csv("index3.csv")
# df4 = pd.DataFrame.from_dict(indexf4,orient='index')
# df4.to_csv("index4.csv")
# print "features index 1"
# print indexf1