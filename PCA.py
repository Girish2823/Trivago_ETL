# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 01:17:37 2018

@author: giris
"""

import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\PCA\Wine.csv')
x = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

#Splitting the dataset into Test and Training Set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)

#Applying PCA
from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#AFter we determine the Top 2 Principal Components so we change
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#explained_variance contains the percentage of variance of all the Principal Components that we created in the earlier step
explained_variance = pca.explained_variance_ratio_
#Since we had 13 independent variables, it has extracted 13 principal components. 
#These,are the new independent variables extracted.
#The, principal components are arranged from the variable with most variance to the least.
#We,select the top 2 principal components as that serves as a good classification model.

#Fitting the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
