# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 00:59:29 2019

@author: priya
"""

import numpy as np
import random
import matplotlib.pyplot as plt

l=[]
for i in range(50):
    l.append([])
    for j in range(2):
        l[i].append(random.randint(1,100))
        
X=np.array(l)

plt.scatter(X[:,0], X[:,1], s=150)
print("initial plot")
plt.show()
colors = 50*["g","r","c","b","k"]
 
class K_means:
    def __init__(self, k=3, tol=0.0001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        self.centroids ={}
        
        for i in range(self.k):
            self.centroids[i]=data[i]
        
        for i in range(self.max_iter):
            self.classifications={}
            
            for i in range(self.k):
                self.classifications[i]=[]
            
            for features in X:
                distances =[np.linalg.norm(features -self.centroids[i]) for i in self.centroids]
                i=distances.index(min(distances))
                self.classifications[i].append(features)
                
            prev_centroids =dict(self.centroids)
            
            for i in self.classifications:
                self.centroids[i]=np.average(self.classifications[i],axis=0)
            
            
            optimized=True
            
            for i in self.centroids:
                original_centroid=prev_centroids[i]
                current_centroid=self.centroids[i]
                
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
                
            
            if optimized:
                break
            
            
    def predict(self,data):
            classifications=[]
            for features in data:
                distances =[np.linalg.norm(features -self.centroids[i]) for i in self.centroids]
                i=distances.index(min(distances))
                classifications.append(i)
            return np.array(classifications)
            
                
clf = K_means()
clf.fit(X)

y_pred=clf.predict(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="y", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="+", color=color, s=150, linewidths=5)
        
plt.show()