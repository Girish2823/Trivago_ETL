# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:32:36 2018

@author: giris
"""

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag

example = "Hello World this is a simple test.Mr. Jack & Ms. Jill went up the hill."

sents = sent_tokenize(example)
#This extracts sentences, i.e. after the fullstop the Text is separated.
print(sents)

#Word Toeknizing this means, we separate out the words from the sentences.
words = word_tokenize(example)
print(words)

#Parts of Speech i.e. Verb base form, Verb past tense etc.
print(nltk.pos_tag(words))