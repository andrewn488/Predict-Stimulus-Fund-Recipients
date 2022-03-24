# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:33:38 2021

@author: ANalundasan
DTC: PCA code
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
# from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

################################# PCA ########################################

misrate_pca = {}
clf = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)

for n in range(len(X[0])):
    pca = PCA(n_components = n+1)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_new = pca.inverse_transform(X_pca)
    
    Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_new)
    
    # create logistic regression and train the model
    clf_pca = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrain, Ytrain)
    
    Ypred_pca = clf.predict(Xtest_pca)
    
    misrate_pca[n+1] = np.sum(np.abs(Ytest_pca - Ypred_pca)) / (2 * len(Ytest))
    
plt.plot(misrate_pca.keys(), misrate_pca.values())
plt.ylabel("Misclassification Rate")
plt.xlabel("Number of Components")
plt.title("Misclassification Rate by Number of Components")
plt.show()
    
