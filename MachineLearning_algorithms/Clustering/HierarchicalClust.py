# -*- coding: utf-8 -*-

#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#Dendograma as cluster number optimization
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram( sch.linkage(X, method = 'ward') )
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclidea')
plt.show()

"""
From the resulting graph we need to choose the optimal cluster number
using the max vertical distance without intercepting any horizontal line
In this case is N = 5
"""
optimal_clusters = 5

#Hierarchy cluster adjust
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = optimal_clusters, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)

#Result plotting
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 10, c = 'red', label = 'cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 10, c = 'blue', label = 'cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 10, c = 'green', label = 'cluster3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 10, c = 'cyan', label = 'cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 10, c = 'magenta', label = 'cluster5')

plt.title('Customers Cluster')
plt.xlabel('Anual income K$')
plt.ylabel('Spent rate (1-100)')
plt.legend()
plt.show()

