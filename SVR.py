# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 00:38:52 2018

@author: giris
"""

#SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing a Dataset
dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\SVR\Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Feauture Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.transform(Y)

#Fitting the SVR Model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting the result
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing the  Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'Blue')
plt.title('Truth or Bluff  SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()