# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:42:39 2018

@author: giris
"""

#Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("E:\\Machine Learning\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Polynomial_Regression\\Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting a Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() #Creating the Linear Model
lin_reg.fit(X,Y) #Fitting the data points on the model

#Fitting a Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_Poly = poly_reg.fit_transform(X)
lin_reg_2= LinearRegression()
lin_reg_2.fit(X_Poly,Y)

#Visualizing the Linear Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'Blue')
plt.title('Truth or Bluff Linear Regresion')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()

#Visualizing the Polynomial Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'Blue')
plt.title('Truth or Bluff Polynomial Regresion')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict(6.5)
#Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))