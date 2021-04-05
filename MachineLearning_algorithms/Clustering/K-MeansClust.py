# -*- coding: utf-8 

#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

#"CODO" method to determine optim cluster number
from sklearn.cluster import KMeans

WCSS = []           #Vector para almacenar los resultados del WCSS
nclusters = 11      #Max cluster number to iterate

for i in range (1, nclusters):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)    #Mean distance add to WCSS vector
    
plt.plot(range(1,11), WCSS)
plt.title('Codo method')
plt.xlabel('Cluster number')
plt.ylabel('WCSS(K)')
plt.show()

optim_cluster = 5

#Dataset Clustering using K-means algorithm

kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Cluster visualization
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, c = 'yellow', label = 'Baricentros')
plt.title('Client clusters')
plt.xlabel('Anual Income')
plt.ylabel('Spent (1-100')
plt.legend()
plt.show()

       
