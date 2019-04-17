# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:39:06 2018

@author: giris
"""

# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 5 - Association Rule Learning\Section 28 - Apriori\Apriori_Python\Market_Basket_Optimisation.csv', header = None)
#Apriori Function is expecting a list of Lists as the input.
#That is a list that contains diffrent transactions each one put in a list.
#Currently, the input is a Data Frame and needs to be converted to a list.
transactions = []
#Building the List of Lists. First, For Loop will loop over all the transactions
#Second For Loop is for all the products in each of the transaction.
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])#Appending different transactions of different customers, value of i is fixed so it corresponds to a specific transaction
    #The products are indexed using j, we use the str function to convert the list to String as that is the input format Apriori function.
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence = 0.2,min_lift = 3,min_length = 2)

# Visualising the results
results = list(rules)