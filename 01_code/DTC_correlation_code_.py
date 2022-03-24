# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:54:30 2021

@author: ANalundasan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.model_selection import train_test_split


# read in data and drop unnecessary columns
data = pd.read_csv('raw_data_numerical.csv', sep = ',')

data = data.drop(['YEAR'], axis = 1)
data = data.drop(['SERIAL'], axis = 1)
data = data.drop(['STATEFIP'], axis = 1)
data = data.drop(['CPSIDP'], axis = 1)
data = data.dropna()

# set X and Y values
X = data.values[:, 0:8]
Y = data.values[:, -1]

# separate train data vs test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

# data exploration
for elem in data: 
    if elem != 'COVIDUNAW': 
        plt.scatter(data[elem], data['COVIDUNAW'])
        plt.title(elem + ' vs. COVIDUNAW')
        plt.show()
        
corr_dict = {}

# correlation test
for feat in data: 
    corr_dict[feat] = [ss.spearmanr(data[feat], data['EMPSTAT'])[0], ss.spearmanr(data[feat], data['COVIDUNAW'])[0]]
    
# del corr_dict['EMPSTAT']
del corr_dict['MONTH']

correlations = []

for elem in corr_dict.values():
    correlations.append(elem[1])
    
y_pos = np.arange(len(correlations))
plt.barh(y_pos, correlations, align='center', alpha=0.5)
plt.yticks(y_pos, corr_dict.keys())
plt.xlabel('Spearman Correlation')
plt.title('Spearman Correlation of features to Unable to Work due to COVID')
plt.show()

for elem in corr_dict:
    print('The P-Value for ', elem, 'is: ', corr_dict[elem][1])


# go with WHYUNEMP, EMPSTAT, AGE, MARST
# Feature selection then use decision tree, and naive bayes

# khadivi to use KNN - 
    # X = [29, M, M, W]
    # Y = 35, S, F, W]
    # dist = |35-29| +1 +1 +0
    
# also use clustering



