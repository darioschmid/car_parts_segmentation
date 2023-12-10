import numpy as np
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
#import cv2
from sklearn.cluster import KMeans

data = np.load('/Users/evakailing/Desktop/DL/carseg_data/arrays/orange_3_doors_0095.npy')

new_data = np.delete(data,3,2)

img=new_data.reshape((new_data.shape[1]*new_data.shape[0],3))

kmeans=KMeans(n_clusters=5)
s=kmeans.fit(img)

labels=kmeans.labels_
print(labels)
labels=list(labels)

centroid=kmeans.cluster_centers_
print(centroid)

percent=[]
for i in range(len(centroid)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
print(percent)


plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
plt.show()

colors = np.array(centroid/255)

print(kmeans.labels_)
#for i in kmeans.labels_:
#  if i==0: 
#    img[i]=colors[0]
#  elif i == 1:
#    img[i]=colors[1]
#  elif i == 2:
#    img[i]=colors[2]
#  elif i == 3:
#    img[i]=colors[3] 
#  elif i == 4:
#    img[i]=colors[4]

#new = kmeans.fit_transform(img)

print(img)
print('done')
print(img[0])
