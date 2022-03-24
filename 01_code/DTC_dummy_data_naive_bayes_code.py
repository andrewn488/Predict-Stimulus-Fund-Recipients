# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:08:07 2021

@author: ANalundasan
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# read in data and drop unnecessary columns
data = pd.read_csv('raw_data_categorical.csv', sep = ',')

# set X and Y values
data = data.drop(['YEAR'], axis = 1)
data = data.drop(['SERIAL'], axis = 1)
data = data.drop(['STATEFIP'], axis = 1)
data = data.drop(['CPSIDP'], axis = 1)
data = data.dropna()
data = pd.get_dummies(data)

# set X and Y values
X = data.values[:, 0:-1]
Y = data.values[:, -1]



# separate train data vs test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

## CATEGORICAL NAIVE BAYES ##
clf = CategoricalNB()
clf.fit(Xtrain, Ytrain)
y_fitted = clf.predict(Xtest)
acc = accuracy_score(Ytest, y_fitted)
print("Categorical Naive Bayes: ", clf.predict(Xtest))
print("Accuracy score is", acc)

# solve for misclassification
misrate = np.sum(np.abs(Ytest-y_fitted))/len(Ytest)
print("Misclassification rate is: ", misrate)

## Try to make a plot ##
COVIDUNAW_yes = [i for i in range(len(y_fitted)) if y_fitted[i]==1]
COVIDUNAW_no  = [i for i in range(len(y_fitted)) if y_fitted[i]==2]
X_yes = X[COVIDUNAW_yes,:]
X_no = X[COVIDUNAW_no,:]


plt.scatter(X_yes[:, 0], X_yes[:, 1], label='COVIDUNAW_yes', c='b')
plt.scatter(X_no[:, 0], X_no[:, 1], label='COVIDUNAW_no', c='r')
plt.legend()
# plt.ylabel("Why Unemployed")
# plt.xlabel("Count")
plt.title("Naive Bayes Classification Plot")
plt.show()