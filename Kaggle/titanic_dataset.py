# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 21:41:02 2019

@author: priya
"""
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
df_train=pd.read_csv('F:/machine learning/titanic/train.csv')
df_train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
l=df_train.isnull().sum()
k=0
for i in range (8):
    if(l[i]>0):
        if(((l[i]/891)*100) > 15):
            df_train.drop(df_train.columns[i-k],axis=1,inplace=True)
            k=k+1
        
train_X=df_train.iloc[:,1:].values
train_y=df_train.iloc[:,0]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb=LabelEncoder();
train_X[:, 5] = lb.fit_transform(train_X[:, 5].astype(str))
lb2=LabelEncoder();
train_X[:, 1] = lb2.fit_transform(train_X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
train_X = onehotencoder.fit_transform(train_X).toarray()
onehotencoder1 = OneHotEncoder(categorical_features = [6])
train_X = onehotencoder1.fit_transform(train_X).toarray()

train_X=train_X[:,1:]
df_test=pd.read_csv('F:/machine learning/titanic/test.csv')

df_test.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
l2=df_test.isnull().sum()
k=0

for i in range (7):
    if(l2[i]>0):
        if(((l2[i]/891)*100) > 10):
            print(df_test.columns[i-k])
            df_test.drop(df_test.columns[i-k],axis=1,inplace=True)
            k=k+1
            l2[i]=0

df_test.drop(['Age'],axis=1,inplace=True)
                   
test_X=df_test.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
test_X[:, 1] = lb.fit_transform(test_X[:, 1])   

lb3=LabelEncoder()  
test_X[:, 5] = lb3.fit_transform(test_X[:, 5].astype(str))

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN" , strategy="median",axis=0)
imputer=imputer.fit(test_X[:,:])
test_X[:,:]=imputer.transform(test_X[:,:])


onehotencoder2 = OneHotEncoder(categorical_features = [1])
test_X = onehotencoder2.fit_transform(test_X).toarray()
onehotencoder3 = OneHotEncoder(categorical_features = [6])
test_X = onehotencoder3.fit_transform(test_X).toarray()

#fitting KNN on dataset 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(train_X,train_y)
y_pred=knn.predict(test_X)

df3=pd.read_csv('F:/machine learning/titanic/sub.csv')
y_sub=df3.iloc[:,1].values
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_sub, y_pred)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(train_X)
X_test=sc.fit_transform(test_X)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, train_y)
y_pred2 = classifier.predict(X_test)
cm2 = confusion_matrix(y_sub, y_pred2)
