# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:39:28 2018

@author: giris
"""

#Decision Tree 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 8 - Decision Tree Regression\Decision_Tree_Regression\Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting the Decison Tree Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

#Predicting a new result
y_pred = regressor.predict(6.5)

#Visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()