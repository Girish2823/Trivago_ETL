# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:46:11 2018

@author: giris
"""

from nltk.corpus import PlaintextCorpusReader

#defining the Corpus Directory
dirname_politics = "E:\\NLP\Datsets\\mini_newsgroups\\talk.politics.misc"
#Reading the data with Corpus
politics_corpus  = PlaintextCorpusReader(dirname_politics,'.*')

#All File Names in the Directory
politics_corpus.fileids()

#Reading the news fields
politics_corpus.raw('176869')
politics_corpus.raw('176878')

#Reading the words in the News Articles
politics_corpus.words('176869')[1:100]

#Importing the data
import pandas as pd
user_restaurants_reviews = pd.read_csv("E:\\NLP\\Datsets\\User_Reviews\\User_restaurants_reviews.csv")
user_restaurants_reviews.shape
user_restaurants_reviews.head(20)

#Taking a small dataset
user_data_tiny = user_restaurants_reviews[0:3]
user_data_tiny
user_data_tiny.columns.values

#Tokenizing
from nltk.tokenize import sent_tokenize,word_tokenize

example_text = user_data_tiny["Review"][0]
print(example_text)

sent_tokens = sent_tokenize(example_text)
print(sent_tokenize)

word_tokens = word_tokenize(example_text)
print(word_tokenize)

#Stop Words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(len(stop_words))
print(stop_words)

#Removing the Stop Words
filtered_sentence = [word for word in word_tokens if not word in stop_words]
print(filtered_sentence)

"""
The expanded version of above code, which is a short-hand notation.
filtered_sentence1 = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence1.append(w)
print(filtered_sentence1)
"""

##########################Update with your own stop words
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) 
print(len(stop_words))
print(stop_words)

filtered_sentence1 = [word for word in word_tokens if not word in stop_words] 
print(filtered_sentence1)

#####################Stemming
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()	#Defining the Stemmer

#Stemming works better if we tokenize the sentences first.
example_text1 = user_data_tiny["Review"][1]
print(example_text1)

word_tokens1 = word_tokenize(example_text1)
print(word_tokens1)

stem_tokens=[stemmer.stem(word) for word in word_tokens1]
print(stem_tokens)

#########################Lemmatizing
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 			#Choosing the Lemmatizer
Lemmatized_tokens = [lemmatizer.lemmatize(word) for word in word_tokens1] 
print(Lemmatized_tokens)

#lemmatised vs Stemmed Tokens
print(Lemmatized_tokens)
print(stem_tokens)

#########################RegEx

review22_text = user_restaurants_reviews["Review"][22]

import re
#re.sub(regexpattern, replacement, string)
#Replacing numbers and currency with space
review22_text_cleaned=re.sub(r'\W+|\d+|_',  ' ',  review22_text)
print("Text after removing currency - \n " + review22_text_cleaned)

print("Actual Text - \n " + review22_text)

########################################
#Case Study : The news articles text mining
###########################################

import os
path = "E:\\NLP\Datsets\\mini_newsgroups\\sci.space"
doc_dict = {}

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        f = open(file_path, 'r')
        text = f.read()
        doc_dict[file] = text

#########lowering all the cases
for my_var in doc_dict:
    doc_dict[my_var] = doc_dict[my_var].lower()

############Removing Numbers
# Use regular expressions to do a find-and-replace

for my_var in doc_dict:
    doc_dict[my_var] = re.sub(r'\d+',           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      doc_dict[my_var])  # The text to search

##################################Removing Punctuations
# Use regular expressions to do a find-and-replace
for my_var in doc_dict:
    doc_dict[my_var] = re.sub(r'\W+|\_',           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      doc_dict[my_var])  # The text to search


###################Removing General English Stop words
from nltk.corpus import stopwords
from nltk import word_tokenize
stop = stopwords.words('english')
for my_var in doc_dict:
    doc_dict[my_var] = ' '.join([i for i in word_tokenize(doc_dict[my_var]) if i not in stop])

################Removing custom stop words
custom_stop = stopwords.words('english') + ['news', 'writes', 'told']
for my_var in doc_dict:
    doc_dict[my_var] = ' '.join([i for i in word_tokenize(doc_dict[my_var]) if i not in custom_stop])

############## Stemming ; This step can be ignored
doc_dict_1=doc_dict
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for my_var in doc_dict_1:
    doc_dict_1[my_var] = ' '.join([stemmer.stem(i) for i in word_tokenize(doc_dict_1[my_var])])

############## Lemmatising

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 			#Choosing the Lemmatizer
for my_var in doc_dict:
    doc_dict[my_var] = ' '.join([lemmatizer.lemmatize(i) for i in word_tokenize(doc_dict[my_var])])
doc_dict['60804']