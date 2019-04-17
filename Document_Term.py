# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:59:57 2018

@author: giris
"""

#####################################################################
###Document Trem Matrix

#Data Import
import pandas as pd
User_restaurants_reviews = pd.read_csv("E:\\NLP\\Datsets\\User_Reviews\\User_restaurants_reviews.csv")
User_restaurants_reviews.shape
User_restaurants_reviews.head(20)

##############
#Lets take a small data, we will work on complete dataset later
input_data = User_restaurants_reviews[0:3]

#Creating Document Term Matrix

from sklearn.feature_extraction.text import CountVectorizer

countvec1 = CountVectorizer()
dtm_v1 = pd.DataFrame(countvec1.fit_transform(input_data['Review']).toarray(), columns=countvec1.get_feature_names(), index=None)
dtm_v1.head()

#####################################################
###Larger DTM

user_data_r100 =User_restaurants_reviews[0:100]

countvec1 = CountVectorizer()
Test_DTM_r100 = pd.DataFrame(countvec1.fit_transform(user_data_r100['Review']).toarray(), columns=countvec1.get_feature_names(), index=None)
Test_DTM_r100.head()