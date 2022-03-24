# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:49:11 2021

@author: ANalundasan
DTC - Categorical Naive Bayes
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# read in data and drop unnecessary columns
data = pd.read_csv('raw_data_numerical_target_features.csv', sep = ',')

# set X and Y values
X = data.values[:, 0:4]
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



## GAUSSIAN NAIVE BAYES ##
# gnb = GaussianNB()
# gnb.fit(Xtrain, Ytrain)


# model = GaussianNB()
# model = CategoricalNB()
# model.fit(Xtrain, Ytrain)
# y_fitted = model.predict(Xtest)

# colors=np.array(["red", "blue"])
# # plt.scatter(X, color=colors[y_fitted])
# plt.scatter([Ytrain], label='Train', c='b')
# # plt.scatter([Ytrain], [y_fitted], label='Train', c='r')
# plt.legend()
# plt.show()

# plt.scatter([Xtest], [Ytest], color="red", label="0")
# plt.scatter([Ytrain], [y_fitted], color="blue", label="1")

