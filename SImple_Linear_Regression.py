# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:03:31 2018

@author: giris
"""

#Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression\Salary_Data.csv")
dataset
X = dataset.iloc[:,:-1].values #Matrix of Independant Variables
Y = dataset.iloc[:,1].values #Vector of Dependant Varibales
#Spliting it in Test and Train Datatset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/2, random_state = 0)

#Fitting the Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #Function that will return an object of itself.
regressor.fit(X_train,Y_train) #Regressor learns the corelation of the training set, i.e. the model has established the existing correlation between Salary and Experience.
 
#Predicting the Test set results
#Will create a Vector of predicted salaries i.e. the dependant variables
y_pred = regressor.predict(X_test)

#Visualizing the Training Set
plt.scatter(X_train,Y_train,color = 'red') #A Scatter plot representing the actual distribution of Salary and Years of Experience on the basis of Training Set.
plt.plot(X_train,regressor.predict(X_train),color = 'blue') # Plots the straight line which is of predicted values, on the basis of the Independent Variables.
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylable('Salary')
plt.show()

#Visualizing the Test Set
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #Don't need to change the regressor line, as the regressor has been trained on the trianing set, if we change it to X_test we will just obtain few nwe points.
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()