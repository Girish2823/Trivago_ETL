# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:55:50 2018

@author: giris
"""

#Reading the files as corpus
import nltk
from nltk.corpus import PlaintextCorpusReader

doc_dirname_politics = "E:\\NLP\\Datsets\\mini_newsgroups\\talk.politics.misc"
doc_dirname_comps = "E:\\NLP\\Datsets\\mini_newsgroups\\comp.os.ms-windows.misc"

politics_news_corpus = PlaintextCorpusReader(doc_dirname_politics,'.*')
politics_news_corpus.fileids()
comp_news_corpus = PlaintextCorpusReader(doc_dirname_comps, '.*')

###############
##Preprocessing our corpus into documents
import re
from nltk.stem.porter import PorterStemmer
#from nltk.corpus import stopwords
#####Writing a Custom TOkenizer
stemmer = PorterStemmer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def custom_preprocessor(text):
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations
    text = nltk.word_tokenize(text)       #tokenizing
    text = [word for word in text if not word in stop_words] #English Stopwords
    text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
    return text

############
#testing our preprocessor
#custom_preprocessor(politics_news_corpus.raw('179097'))

politics_news_docs = [(custom_preprocessor(politics_news_corpus.raw(fileid)), 'politics')
                for fileid in politics_news_corpus.fileids()]
politics_news_docs[0]

comp_news_docs = [(custom_preprocessor(comp_news_corpus.raw(fileid)), 'comp')
                for fileid in comp_news_corpus.fileids()]
comp_news_docs[0]


############
#Merging our corpus into single documents object
documents = politics_news_docs + comp_news_docs
import random
random.seed(50)
random.shuffle(documents)

documents[0]
documents[1]
#################
#Building featuresets
#Building NLTK friendly Document Term Matrix(generally called featuresets)
all_words = []
for t in documents:
    for w in t[0]:
        all_words.append(w)
        
all_words_freq = nltk.FreqDist(all_words)
print(all_words_freq.most_common(150))
print(all_words_freq['president'])

#word_features = list(all_words_freq.keys())[:3000]
word_features = all_words
word_features
##defining a function to map our NLTK friendly dataset
##defining a function to map our NLTK friendly dataset
def find_features(docs):
    words = set(docs)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#print((find_features(politics_news_corpus.words('176878'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets[1:5]

#################
#Creating training and testing data

train_set = featuresets[:160]
test_set = featuresets[160:]

#Building Naive Bayes Model
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('accuracy is : ', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(25)

#################
#Creating training and testing data

train_set = featuresets[:160]
test_set = featuresets[160:]

#Building Naive Bayes Model
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('accuracy is : ', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(25)