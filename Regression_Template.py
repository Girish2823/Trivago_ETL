# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:26:46 2018

@author: giris
"""

#Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("E:\\Machine Learning\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Polynomial_Regression\\Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values



#Fitting the Regression Model





#Predicting a new result with Polynomial Regression
y_pred =regressor.predict(6.5)



#Visualizing the  Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'Blue')
plt.title('Truth or Bluff  Regresion')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()