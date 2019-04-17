# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Thu Jul 26 16:05:21 2018)---
runfile('C:/Users/giris/.spyder-py3/KNN.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Thu Jul 26 16:13:57 2018)---
runfile('C:/Users/giris/.spyder-py3/KNN.py', wdir='C:/Users/giris/.spyder-py3')
import scipy.sparse.linalg
import numpy

## ---(Thu Jul 26 16:43:34 2018)---
import scipy
runfile('C:/Users/giris/.spyder-py3/KNN.py', wdir='C:/Users/giris/.spyder-py3')
import scipy
import numpy
import pandas
import matplotlib
runfile('C:/Users/giris/.spyder-py3/KNN.py', wdir='C:/Users/giris/.spyder-py3')
import iterative.py

## ---(Thu Jul 26 20:33:15 2018)---
runfile('C:/Users/giris/.spyder-py3/KNN.py', wdir='C:/Users/giris/.spyder-py3')
import theano
import tensorflow

## ---(Tue Jul 31 14:00:16 2018)---
runfile('C:/Users/giris/.spyder-py3/SVR_Classification.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Mon Aug  6 00:12:07 2018)---
runfile('C:/Users/giris/.spyder-py3/Kernel_SVM.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Tue Aug  7 22:25:54 2018)---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 17 - Kernel SVM\Kernel_SVM\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
classifier.fit(X_train,y_train)

#Predicting thet test set result
y_pred = classifier.predict(X_test)
runfile('C:/Users/giris/.spyder-py3/Kernel_SVM.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Thu Aug 23 14:04:22 2018)---
runfile('C:/Users/giris/.spyder-py3/naive_bayes.py', wdir='C:/Users/giris/.spyder-py3')
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
runfile('C:/Users/giris/.spyder-py3/naive_bayes.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Sat Aug 25 12:57:00 2018)---
runfile('C:/Users/giris/.spyder-py3/Decision_Tree_Classification.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Aug 31 08:47:56 2018)---
runfile('C:/Users/giris/.spyder-py3/random_forest_classification.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Sat Sep  8 11:44:56 2018)---
runfile('C:/Users/giris/.spyder-py3/Trivago.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Sep 14 11:18:19 2018)---
import numpy as np
import matplotlib as plt
import pandas as pd
dataset = import.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\K_Means\Mall_Customers.csv')
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\K_Means\Mall_Customers.csv')
X = dataset.iloc[:,3:4]
X = dataset.iloc[:,3:5]
X = dataset.iloc[:,3:5].values
from sklearn.cluster import KMeans

## ---(Sat Sep 15 09:01:13 2018)---
runfile('C:/Users/giris/.spyder-py3/KNN_clustering.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Sep 21 16:45:49 2018)---
runfile('C:/Users/giris/.spyder-py3/Basics.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/scrapping.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Mon Sep 24 14:29:02 2018)---
runfile('C:/Users/giris/matches.py', wdir='C:/Users/giris')
debugfile('C:/Users/giris/matches.py', wdir='C:/Users/giris')
runfile('C:/Users/giris/matches.py', wdir='C:/Users/giris')
runfile('C:/Users/giris/grid.py', wdir='C:/Users/giris')
runfile('C:/Users/giris/matches.py', wdir='C:/Users/giris')

## ---(Wed Oct  3 00:22:21 2018)---
runfile('C:/Users/giris/jsonprocessing.py', wdir='C:/Users/giris')

## ---(Sun Oct  7 11:09:12 2018)---
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')
soup.find_all('table',class_ = 'table table-condensed')
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')
type(info_table)
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')
type(info_table)
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')

## ---(Sun Oct  7 18:51:11 2018)---
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')

## ---(Mon Oct  8 00:14:28 2018)---
n = int(raw_input())
n = int(input())
w = len("{0:b}".format(n))
print w
print(w)
print("{0:{width}d}".format(1,w))
print("{0:{w}d}".format(1,w))
print("{0:{width}d}".format(1,width=w))
print("{0:{width}b}".format(1,width=w))
str = "This article is written in {}"
print(str.format("Python"))
print("Hello, I am {} years old 1".format(18))

## ---(Tue Oct  9 14:44:41 2018)---
runfile('C:/Users/giris/re.py', wdir='C:/Users/giris')
string = "Hello World"
string.split()
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')
print(info_table)
runfile('C:/Users/giris/Web_scrapping.py', wdir='C:/Users/giris')
print(info_table)

## ---(Wed Oct 10 00:34:03 2018)---
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')

## ---(Wed Oct 10 19:34:27 2018)---
runfile('E:/AIT580-Prof/basics.py', wdir='E:/AIT580-Prof')

## ---(Sat Oct 13 00:04:57 2018)---
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')

## ---(Sat Oct 13 10:01:33 2018)---
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')
review
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')
"""
Created on Wed Oct 10 00:35:03 2018

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

#Compulsory step to clean the text....

#Will be removing puncation marks, numbers, and words which do not give serve any prupose for Text analysis
#like The,and,or etc.
#We will also use Stemming which reduces past tense words to simple form example loved to love, liked to like etc.
#Get rid of capitals review contains lower case words
#Create a Bag of words model
#Will create a Sparse Matrix.
corpus = []

#Cleaning the Text
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    #Stemming keepin the Root of Word only
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')

## ---(Sat Oct 13 14:37:49 2018)---
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')

## ---(Fri Oct 19 08:39:19 2018)---
runfile('C:/Users/giris/.spyder-py3/data_preprocessing_template.py', wdir='C:/Users/giris/.spyder-py3')
import pandas as pd
import numpy as np
import matplotlib as plt

runfile('C:/Users/giris/data_work.py', wdir='C:/Users/giris')
runfile('C:/Users/giris/.spyder-py3/data_preprocessing_template.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Mon Oct 22 01:08:42 2018)---
runfile('C:/Users/giris/.spyder-py3/PCA.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Tue Oct 23 23:25:31 2018)---
runfile('C:/Users/giris/.spyder-py3/PCA.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Wed Oct 24 22:54:39 2018)---
runfile('C:/Users/giris/.spyder-py3/PCA.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Oct 26 13:05:47 2018)---
runfile('C:/Users/giris/.spyder-py3/data_preprocessing_template.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/data_work.py', wdir='C:/Users/giris')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

runfile('C:/Users/giris/data_work.py', wdir='C:/Users/giris')

## ---(Fri Oct 26 19:33:17 2018)---
runfile('C:/Users/giris/.spyder-py3/linear_search.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/binary_search.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/Bubble_Sort.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
dataset.info
dataset.isnull()
dataset.isnull().sum()
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
imputer.strategy
imputer.fit([['Age','Salary']])
imputer.fit(dataset[['Age','Salary']])
imputer.statistics_
imputer.statistics_[0]
imputer.statistics_[1]
dataset['Age'].mean()
dataset['Salary'].mean()
dataset
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
imputer.transform(dataset[['Age','Salary']])
dataset
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
encoder = LabelEncoder()
encode.classes_
encode.fit(dataset['Country'])
encode.classes_
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
encode.transform(dataset['Country'])
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
x1
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
x1
import numpy as np

x1 = np.array([1,2,3],
              [4,5,6],
              [7,8,9])
              
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
standardscaler.mean_
standardscaler.var_
standardscalar.scale_
standardscaler.scale_
np.std(x1[0])
x1[0]
np.std(x1[:,0])
standardscalar.fit(x1)
standardscaler.fit(x1)
standardscaler.scale_
standardscaler.get_params

## ---(Sat Oct 27 13:54:29 2018)---
testlist = random.sample(range(0,10),10)

import random

testlist = random.sample(range(0,10),10)

print(testlist)

## ---(Sat Oct 27 22:05:50 2018)---
runfile('C:/Users/giris/.spyder-py3/quick_sort.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Sat Oct 27 23:21:01 2018)---
runfile('C:/Users/giris/.spyder-py3/Stacks.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/PCA.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Nov  2 12:55:28 2018)---
runfile('C:/Users/giris/.spyder-py3/multiple_regression.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/Pandas_basics.py', wdir='C:/Users/giris/.spyder-py3')
imputer.statistics_
imputer.transform
imputer.statistics_[0]
imputer.strategy
df['Age'].mean()
dataset['Age'].mean()
runfile('C:/Users/giris/.spyder-py3/SImple_Linear_Regression.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/multiple_regression.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Mon Nov 12 10:30:16 2018)---
runfile('C:/Users/giris/.spyder-py3/Sentiment.py', wdir='C:/Users/giris/.spyder-py3')
nltk.download
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')
nltk.download()
runfile('C:/Users/giris/.spyder-py3/PCA.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/Sentiment.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/NLP.py', wdir='C:/Users/giris')
cm
runfile('C:/Users/giris/.spyder-py3/Sentiment.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Tue Nov 13 08:28:00 2018)---
runfile('C:/Users/giris/.spyder-py3/K-Fold.py', wdir='C:/Users/giris/.spyder-py3')
accuracies.mean()

## ---(Sat Nov 17 15:02:41 2018)---
runfile('C:/Users/giris/.spyder-py3/NLP_Work.py', wdir='C:/Users/giris/.spyder-py3')
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review
runfile('C:/Users/giris/.spyder-py3/NLP_Work.py', wdir='C:/Users/giris/.spyder-py3')
review
runfile('C:/Users/giris/.spyder-py3/NLP_Work.py', wdir='C:/Users/giris/.spyder-py3')
review
runfile('C:/Users/giris/.spyder-py3/NLP_Work.py', wdir='C:/Users/giris/.spyder-py3')
y_predict
y_pred

## ---(Mon Nov 26 00:22:39 2018)---
runfile('C:/Users/giris/.spyder-py3/grid_search.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Wed Nov 28 21:25:50 2018)---
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
tokens[0:]
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
words[0:]
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
sw
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
words_ns
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
nlp_words
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
nltk.download('all')
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
politics_corpus.fileids()
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
politics_corpus.raw('176869')
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
politics_corpus.words('176869')[1:100]
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
user_restaurants_reviews.shape
user_restaurants_reviews.head(20)
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
user_data_tiny
user_data_tiny.columns.values
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Thu Nov 29 23:41:35 2018)---
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
print(doc_dict)
print(doc_dict.keys())
doc_dict['62480']
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
doc_dict.keys()
doc_dict['62480']
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
doc_dict.keys()
doc_dict['62480']
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Nov 30 10:36:55 2018)---
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
doc_dict.keys()
doc_dict['62477']
runfile('C:/Users/giris/.spyder-py3/NLP_Text_Mining.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/Document_Term.py', wdir='C:/Users/giris/.spyder-py3')
dtm_v1.head()
runfile('C:/Users/giris/.spyder-py3/Document_Term.py', wdir='C:/Users/giris/.spyder-py3')
Test_DTM_r100.head()
runfile('C:/Users/giris/.spyder-py3/Sentiment_Analysis.py', wdir='C:/Users/giris/.spyder-py3')
input_data['class'].value_counts()
input_data.head(10)
runfile('C:/Users/giris/.spyder-py3/Sentiment_Analysis.py', wdir='C:/Users/giris/.spyder-py3')
runfile('E:/AIT580-Prof/nlp-demo.py', wdir='E:/AIT580-Prof')
nltk.FreqDist(corpus)
gutenberg.words()
freq_dist.most_common(20)
freq_dist.plot(20)
runfile('C:/Users/giris/.spyder-py3/Document_Categorisation.py', wdir='C:/Users/giris/.spyder-py3')
politics_news_corpus.fileids()
politics_news_corpus.raw('179097')
runfile('C:/Users/giris/.spyder-py3/Document_Categorisation.py', wdir='C:/Users/giris/.spyder-py3')
runfile('C:/Users/giris/.spyder-py3/word_count.py', wdir='C:/Users/giris/.spyder-py3')
d.keys()
print(d[w])
for w in d:
    d[w] = d[w].lower()
    
filer
file
text = file.read
text = file.read()
f = open(file,'r')
text = f.read()
d = 
doc = {}
doc[file]=text
doc
doc.keys()
path = ""E:\\NLP\\Datsets\\WHO_Report.txt"
path = "E:\\NLP\\Datsets\\WHO_Report.txt"
for file in os.walk(path):
    f = open(path,'r')
    text = read.f()
    doc[file] = text
    
import os
for file in os.walk(path):
    f = open(path,'r')
    text = read.f()
    doc[file] = text
    
doc[file]
doc[file].lower()
text
import re
t = re.findall('\w+',text)
t[0:]
w = []
for w in t:
    w.append(w.lower())
    
w = []
for z in t:
    w.append(z.lower())
    
w[0:]
sw = nltk.corpus.stopwords('english')
runfile('C:/Users/giris/.spyder-py3/NLP_basics_work.py', wdir='C:/Users/giris/.spyder-py3')
tokens[0:]
runfile('C:/Users/giris/.spyder-py3/NLP_basics_work.py', wdir='C:/Users/giris/.spyder-py3')
print(di)
runfile('C:/Users/giris/.spyder-py3/NLP_basics_work.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Fri Nov 30 18:18:16 2018)---
runfile('C:/Users/giris/.spyder-py3/NLP_basics_work.py', wdir='C:/Users/giris/.spyder-py3')
corpus = gutenberg.words()
corpus
freq = nltk.FreqDist(corpus)
print(freq)
runfile('C:/Users/giris/.spyder-py3/NLP_basics_work.py', wdir='C:/Users/giris/.spyder-py3')
politics_corpus.fileids()
politics_corpus.raw('176869')
politics_corpus.words('176869')[1:100]
#Importing the data
import pandas as pd
user_restaurants_reviews = pd.read_csv("E:\\NLP\\Datsets\\User_Reviews\\User_restaurants_reviews.csv")


## ---(Sun Dec  2 23:30:17 2018)---
runfile('C:/Users/giris/.spyder-py3/Document_Categorisation.py', wdir='C:/Users/giris/.spyder-py3')
word_features
runfile('C:/Users/giris/.spyder-py3/Document_Categorisation.py', wdir='C:/Users/giris/.spyder-py3')

## ---(Wed Dec  5 15:38:21 2018)---
runfile('C:/Users/giris/.spyder-py3/Apriori.py', wdir='C:/Users/giris/.spyder-py3')
3*7/7500
runfile('C:/Users/giris/.spyder-py3/Apriori.py', wdir='C:/Users/giris/.spyder-py3')