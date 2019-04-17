# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:08:50 2018

@author: giris
"""

import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#nltk.download('stopwords')
dataset = pd.read_csv("E:\Machine Learning\Machine Learning A-Z Template Folder\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\\Natural_Language_Processing\\Restaurant_Reviews.tsv",delimiter='\t',quoting = 3)


corpus = []
#Cleaning the Text

for i in range(0,1000):
      
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    #review = [word for word in review if not word in set(stopwords.words('english'))] 
    #removed Stopwords earlier.
    ps = PorterStemmer() #Crating an abject/instance, so that the stem() function can be used.
    #Stemming 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)


#Creating the Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #Crating the Count Vectorizer Class.
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Creating a Training and Test Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
   
# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)      
    





