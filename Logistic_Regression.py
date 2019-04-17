# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:32:44 2018

@author: giris
"""

#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 14 - Logistic Regression\Logistic_Regression\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Splitting into Test and Training Set
from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting the model on training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Visualizing the Training Set results
