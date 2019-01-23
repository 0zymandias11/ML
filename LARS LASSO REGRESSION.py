


# LARS LASSO REGRESSION

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoLars

filename = 'bengin_traffic.csv'
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:114]
y = array[:,114]
kfold = KFold(n_splits=10, random_state=7)

model = LassoLars()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print(results)
print(results.mean())

