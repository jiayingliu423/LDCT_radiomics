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

#%% Demo to test the trained model developed for the pipeline only SS features
# Number of features, feature selection method and classifier
# 5, mrmr, AdaBoost

#%%Load the nodules from the LDCT to test the model, the features already went through feature selection and normalization
test_df = pd.read_csv('test.csv')
X_test = test_df.iloc[:, :-1]  # Select all rows and all columns except the last one
y_test = test_df.iloc[:, -1]  # Select all rows and the last column

# Print the feature names
print("Feature Names:", list(X_test.columns))

# Import the model
model_filename = 'mrmr_Adaboost.pkl'
with open(model_filename, 'rb') as f:
  # Load the model object from the file
  loaded_model = pickle.load(f)

# Optimal_threshold obtained from the validation set
optimal_threshold = 0.4827030042225229
#%% Select a specific instance from the test set to predict
row_index_to_predict = 1  
X_single_instance = X_test.iloc[[row_index_to_predict]]
X_single_instance_array = X_single_instance.values

# Predict probabilities for the selected instance
y_prob = loaded_model.predict_proba(X_single_instance_array)

# Make predictions based on the threshold
y_pred = np.where(y_prob[:, 1] < optimal_threshold, 0, 1)

# Compare with the actual label if needed
actual_label = y_test.iloc[row_index_to_predict]

# Print the results
print("Predicted Class:", y_pred)
print("Actual Class:", actual_label)

#%% Predict all the instances from the test set and evaluate
# Obtain the y_prob
y_probs = loaded_model.predict_proba(X_test.values)

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
