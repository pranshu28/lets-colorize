import ftr_extr_search as ftr
import argparse
import numpy as np
from PIL import Image
import pandas as pd
# from skimage.measure import compare_ssim as ssim
#Argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
args = vars(ap.parse_args())

#extract features for query image
f1,f2,f3,f4 = ftr.main(args["query"])
# f1 = f1.astype(float)
# f2 = f2.astype(float)
# f3 = f3.astype(float)
# f4 = f4.astype(float)

#Read the index files
df1 = pd.read_csv('index1.csv',header=None)
df2 = pd.read_csv('index2.csv',header=None)
df3 = pd.read_csv('index3.csv',header=None)
df4 = pd.read_csv('index4.csv',header=None)


#Function to calculate best match
def result(f,df):
	best = 0							#Use best = float("inf")  while using ssim
	best_src = ""
	cols = df.columns
	for each,row in df.iterrows():
		vector = np.array(row[cols[1:]])
		# vector = vector.astype(float)
		key = str(row[cols[0]])
		inner = np.inner(f,vector)		#Use inner product of the two vectors
		if inner>=best:
			best = inner
			best_src = key
	return best_src

best1 = result(f1,df1)
best2 = result(f2,df2)
best3 = result(f3,df3)
best4 = result(f4,df4)

result1 = Image.open(best1)
result1.show()
print result1
result2 = Image.open(best2)
result2.show()
print result2
result3 = Image.open(best3)
result3.show()
print result3
result4 = Image.open(best4)
result3.show()
print result4
