# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:47:10 2018

@author: girish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creating a Series
s = pd.Series()
print(s)

#Series can be designed from Arrays, Dictionary and Scalar
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[1,2,3,4])
print(s)

#Creating a Series from a Scalar
s = pd.Series(5)
print(s)

#Creating a DataFrame
data = {'Name':['Tom','Harry','Jen','Ben'],'Age':[28,24,26,22]}
df = pd.DataFrame(data,index=['rank1','rank2','rank3','rank4'])
print(df)

#Selecting a Column from the DataFrame
print("********************************")
print(df['Age'])

#Adding a Column to the DataFrame
print("*********************************")
df['University'] = pd.Series(['GMU','UNCC','NCSU'],index=['rank1','rank2','rank3'])
print(df)

#Slicing loc() and iloc() functions
print("**********************************")
print(df.loc['rank1'])

#Selection the All Rows for specific Columns
print("**************************************")
print(df.loc[:,'Age']) 


#Selecting All Rows for Multiple Columns
print("**************************************")
print(df.loc[:,['Name','Age']])

#Selecting Multiple Rows for One Column
print("**************************************")
print(df.loc[['rank1','rank2'],['Name']])

#iloc() function
print("**************************************")
print(df.iloc[0:2,0:1])

#Importing a .csv file into python
dataset = pd.read_csv('E:\AIT580-Prof\movies.csv')

print(dataset.describe())

#Elementary Functions in Pandas
print(dataset.head(5))
print("**************************")
print(dataset.tail(7))
print("**************************")
print(dataset.size)
print("**************************")
print(dataset.dtypes) 

#Sorting by values using sort_values() function
sorted_dataset = dataset.sort_values(by='Release Date')
print(sorted_dataset)

#dropna() and isnull() functions 

print("***********************************")
print(dataset.isnull()) #Checks for the entire dataset

print("***********************************")
#Creating a Subset.....
df = dataset.loc[dataset['IMDB Votes'] > 100]
print(df)

#Visualizing the BarChart
objects = ('Python','Java','C','C++','Scala','Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]
plt.bar(y_pos,performance,align='center',alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel("Usage")
plt.title("Programming Language Usage")
plt.show()