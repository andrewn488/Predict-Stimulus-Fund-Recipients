# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:18:13 2021

@author: ANalundasan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# read in data and drop unnecessary columns
data = pd.read_csv('raw_data_categorical.csv', sep = ',')

data = data.drop(['YEAR'], axis = 1)
data = data.drop(['SERIAL'], axis = 1)
data = data.drop(['STATEFIP'], axis = 1)
data = data.drop(['CPSIDP'], axis = 1)
data = data.dropna()

# set X and Y values
X = data.values[:, 0:8]
Y = data.values[:, -1]

# separate train data vs test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

####### create logistic regression object and train the model ################
clf = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)
misrate = np.sum(np.abs(Ytest - Ypred)) / (2 * len(Ytest))

print("Logistic Regression misclassification rate: %.2f" % misrate)