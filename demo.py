#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:39:38 2024

@author: liujiaying
"""

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.metrics import  confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
# Set the random seed for NumPy
np.random.seed(123)
# Set the random seed for Python's built-in random module
random.seed(123)

#%% Demo to test the trained model developed for the pipeline excluding SS features
# Number of features, feature selection method and classifier
# 20, sfm_GDBT, XGB

#%%Load the nodules from the LDCT to test the model, the features already went through feature selection and normalization
test_df = pd.read_csv('test.csv')
X_test = test_df.iloc[:, :-1]  # Select all rows and all columns except the last one
y_test = test_df.iloc[:, -1]  # Select all rows and the last column

# import the model
model_filename = 'rfe_KNN.pkl'
with open(model_filename, 'rb') as f:
  # Load the model object from the file
  loaded_model = pickle.load(f)

# Obtain the y_prob
y_probs = loaded_model.predict_proba(X_test)
# optimal_threshold obtained from the validation set
optimal_threshold = 0.4
# result
y_pred = np.where(y_probs[:, 1] < optimal_threshold, 0, 1)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs[:, 1])

print('Sensitivity (Recall): %.4f' % sensitivity)
print('Specificity: %.4f' % specificity)
print('Accuracy: %.4f' % accuracy)
print('AUC: %.4f' % auc)
print('Balanced_acc: %.4f' % balanced_accuracy)
print('\nConfusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
