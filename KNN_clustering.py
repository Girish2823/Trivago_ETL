# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:12:20 2018

@author: giris
"""

"KNN Clustering"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the mall dataset
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\K_Means\Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

#Using the Elbow method to determine the number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 300,n_init = 10, random_state = 0)
y_means = kmeans.fit_predict(X)
 
#Visualizing the Cluster
plt.scatter(X[y_means == 0,0],X[y_means == 0,1], s = 100, color = 'red', label ='Careful')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1], s = 100, color = 'blue', label ='Standard')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1], s = 100, color = 'green', label ='Target')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1], s = 100, color = 'cyan', label ='Careless')
plt.scatter(X[y_means == 4,0],X[y_means == 4,1], s = 100, color = 'magenta', label ='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300, c= 'yellow', label = 'centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()