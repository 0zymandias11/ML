
# coding: utf-8

# In[1]:


# BAYESIAN REGRESSION
# using sklearn
# importing required libraries

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge


# In[2]:


# preparing data

filename = 'bengin_traffic.csv'
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:114]
y = array[:,114]
kfold = KFold(n_splits=10, random_state=7)


# In[3]:


# the real deal

model = BayesianRidge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print(results)
print(results.mean())

