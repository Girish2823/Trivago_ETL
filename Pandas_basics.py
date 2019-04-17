# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 02:18:20 2018

@author: giris
"""

import pandas as pd

dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv")
dataset

from sklearn.preprocessing import Imputer, LabelEncoder

imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0) #Creating an Instance of the Imputer Class

#The fit methods calculates the mean as mentioned in the strategy for the Imputer.
imputer.fit(dataset[['Age','Salary']])
#The transform method imputes the mean that was calculated in the places of missing values.
#axis = 0 means column wise imputation
X = imputer.transform(dataset[['Age','Salary']])


#The below step combines the fit and transnform into one step.
imputer.fit_transform(dataset[['Age','Salary']])

############################################
encode = LabelEncoder()
encode.fit(dataset['Country'])
encode.transform(dataset['Country'])

encode.fit_transform(dataset['Country'])

###############################################
import numpy as np
from sklearn.preprocessing import StandardScaler

x1 = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

standardscaler = StandardScaler()

x_scaler = standardscaler.fit_transform(x1)
print(x_scaler)


'''(Xi - Mean) / (Standard Deviation of the feature)'''
'''The mean is calculated column-wise'''
''''standardscaler.scale_ standard deviation of each column''' 
'''standard deviation of 1st column np.std(x1[:0])'''
standardscaler.fit(x1)
standardscaler.transform(x1)

