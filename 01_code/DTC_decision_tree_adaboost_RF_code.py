# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:56:38 2021

@author: ANalundasan
DTC - Decision Tree - Stump, Adaboost, Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# read in data and drop unnecessary columns
data = pd.read_csv('raw_data_numerical_target_features.csv', sep = ',')

# set X and Y values
X = data.values[:, 0:4]
Y = data.values[:, -1]

# separate train data vs test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

# confusion matrix
def print_confusion_matrix(test, pred): 
    """helper function to print out confusion matrix whenever called in 
    subsequent steps of the script"""
    tp, fn, fp, tn = confusion_matrix(test, pred).ravel()
    print("Confusion Matrix: ")
    print("  %9s" % ("Predicted"))
    print("  %4s %4s" % ("Pos", "Neg"))
    print("T %4s %4s" % (tp, fn))
    print("F %4s %4s" % (fp, tn))
    
model = DecisionTreeClassifier(random_state=1)
model.fit(Xtrain, Ytrain)
Y_pred = model.predict(Xtest)
print_confusion_matrix(Ytest, Y_pred)
print(f"Accuracy of Decision Tree Classifier: {accuracy_score(Ytest, Y_pred)}")


# classifier and accuracy score

clf = DecisionTreeClassifier(random_state=1, max_depth=1)
clf.fit(Xtrain, Ytrain)
clf.score(Xtest, Ytest)
y_fitted = clf.predict(Xtest)
acc = accuracy_score(Ytest, y_fitted)
print("Accuracy score is", acc)

# calculate misclassification rate
estimatedY = clf.predict(Xtest)
misrate = np.sum(np.abs(Ytest-estimatedY))/len(Ytest)
print("Misclassification rate is: ", misrate)

########################## WEAK CLASSIFIER ###################################
model = DecisionTreeClassifier(random_state=1, max_depth=1)    # max depth of 1 makes it weak. it's a stump!
model.fit(Xtrain, Ytrain)                # fit the model
Y_pred = model.predict(Xtest)            # make predictions based on test data
print_confusion_matrix(Ytest, Y_pred)    # call helper function to print confusion matrix
print(f"Accuracy of Weak Decision Tree: {accuracy_score(Ytest, Y_pred)}")


########################## ADABOOST ##########################################
for n in [10, 25, 50, 100, 200]:
    print(f"\nAdaboost with n_estimators = {n}")
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1, max_depth=1),
                             n_estimators=n, random_state=1)
    model.fit(Xtrain, Ytrain)                # fit the model
    Y_pred = model.predict(Xtest)            # make predictions based on test data
    print_confusion_matrix(Ytest, Y_pred)    # call helper function to print confusion matrix
    print(f"Accuracy of Adaboost Classifier: {accuracy_score(Ytest, Y_pred)}")

########################## RANDOM FOREST #####################################
for n in [10, 25, 50, 100, 200]:
    print(f"\nRandom Forest with n_estimators = {n}")
    model = RandomForestClassifier(n_estimators=n, random_state=1)
    model.fit(Xtrain, Ytrain)                # fit the model
    Y_pred = model.predict(Xtest)            # make predictions based on test data
    print_confusion_matrix(Ytest, Y_pred)    # call helper function to print confusion matrix
    print(f"Accuracy of Random Forest Classifier: {accuracy_score(Ytest, Y_pred)}")

# Decision Tree plot
fig = plt.figure(figsize=(25,20))
figure = tree.plot_tree(clf, filled=True)

