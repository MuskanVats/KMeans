# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:36:08 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
data=pd.read_csv("Mall_Customers.csv")

X=data.iloc[:,[3,4]].values

#Creating the elbow chart
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#From the graph we get number of clusters =5

#Applying KMeans to Mall Dataset
kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualizing the outcomes
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c="red",label="Careful")
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c="blue",label="Standard")
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c="green",label="Target")
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c="cyan",label="Careless")
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c="magenta",label="Sensible")


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c='yellow', label="Centroids")
plt.title("Cluster of Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
