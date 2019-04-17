# -*- coding: utf-8 -*-
"""
Created on Wed May 30 23:55:57 2018

@author: giris
"""

#Multiple Regression using Backward Elimination

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\\50_Startups.csv")
X = dataset.iloc[:,:-1].values #Indepent Variables
Y = dataset.iloc[:,4].values #Dependent Variables


#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding the Dummy Variable Trap
X = X[:,1:]
#Splitting into Test and Training Set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Fititng the Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)#Fits the regressor to the training set

#Predicting the Test Set
#Since we have 4 Independent and one Dependent variable we cannot plot a graph to see, so we will be predicting the test set result.
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis = 1)
#axis = 1 for line/row and axis = 0 for columns
#Creating a column of 1's because the statsmodels package does not take into account for the constant in regression equation
X_opt = X[:,[0,1,2,3,4,5]]
#Will only contain independent variables which have high impact on the Dependent Variable. 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
#Removing the Attribute with the highest p-value in this case it is column 2
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the Column 1 as the p-value for that is higher than the significance level of 0.05
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the Column 4 as the p-value is gigher than the significance level of 0.05
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the Column 5 as the p-value is higher than the significance level of 0.05
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()