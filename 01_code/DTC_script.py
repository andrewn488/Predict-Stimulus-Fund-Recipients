# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:10:17 2021

@author: ANalundasan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
# from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = pd.read_csv('raw_data_categorical.csv', sep = ',')

data = data.drop(['YEAR'], axis = 1)
data = data.drop(['SERIAL'], axis = 1)
data = data.drop(['STATEFIP'], axis = 1)
data = data.drop(['CPSIDP'], axis = 1)
data = data.dropna()

X = data.values[:, 0:8].astype(str)
Y = data.values[:, -1].astype(str)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

# data exploration
for elem in data: 
    if elem != 'COVIDUNAW': 
        plt.scatter(data[elem], data['COVIDUNAW'])
        plt.title(elem + ' vs. COVIDUNAW')
        plt.show()
        
corr_dict = {}

# correlation test
for col in data: 
    corr_dict[col] = [ss.spearmanr(data[col], data['EMPSTAT'])[0], ss.spearmanr(data[col], data['COVIDUNAW'])[0]]
    
del corr_dict['EMPSTAT']

correlations = []

for elem in corr_dict.values():
    correlations.append(elem[0])
    
y_pos = np.arange(len(correlations))
plt.barh(y_pos, correlations, align='center', alpha=0.5)
plt.yticks(y_pos, corr_dict.keys())
plt.xlabel('Spearman Correlation')
plt.title('Spearman Correlation to Employment Status by Feature')
plt.show()

for elem in corr_dict:
    print('The P-Value for ', elem, 'is: ', corr_dict[elem][1])

####### create logistic regression object and train the model ################
clf = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrain, Ytrain)

Ypred = clf.predict(Xtest).astype(str)

misrate = np.sum(np.abs(Ytest - Ypred)) / (2 * len(Ytest))

print("The misclassification rate with Logistic Regression is: %.2f" % misrate)

# PCA

misrate_pca = {}

for n in range(len(X[0])):
    pca = PCA(n_components = n+1)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_new = pca.inverse_transform(X_pca)
    
    Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_new)
    
    # create logistic regression and train the model
    clf_pca = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrain, Ytrain)
    
    Ypred_pca = clf.predict(Xtest_pca).astype(int)
    
    misrate_pca[n+1] = np.sum(np.abs(Ytest_pca - Ypred_pca)) / (2 * len(Ytest))
    
plt.plot(misrate_pca.keys(), misrate_pca.values())
plt.ylabel("Misclassification Rate")
plt.xlabel("Number of Components")
plt.title("Misclassification Rate by Number of Components")
plt.show()