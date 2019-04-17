# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:44:12 2018

@author: giris
"""

import re
import nltk
import matplotlib.pyplot as plt


text =  """The Senate on Wednesday delivered a historic rebuke of Saudi Arabia and President Trump’s handling of the fallout over journalist Jamal Khashoggi killing last month,
as a decisive majority voted to advance a measure to end U.S. military support for the Saudi-led war in Yemen.
The 63-to-37 vote is only an initial procedural step, 
but it nonetheless represents an unprecedented challenge to 
the security relationship between the United States and Saudi Arabia. 
The vote was prompted by lawmakers’ growing frustration with Trump for 
defending Saudi Crown Prince Mohammed bin Salman’s denials of 
culpability in Khashoggi’s death, despite the CIA’s finding that he had almost certainly ordered the killing.
Their frustration peaked shortly before Wednesday’s vote, when senators met behind closed doors to 
discuss Saudi Arabia, Khashoggi and Yemen with Secretary of State Mike Pompeo and 
Defense Secretary Jim Mattis — but not CIA Director Gina Haspel, who did not attend the briefing.
Her absence so incensed lawmakers that one of the president’s closest congressional 
allies threatened not only to vote for the Yemen resolution but also to withhold 
his support from “any key vote” — including a government funding bill — until Haspel 
was sent to Capitol Hill for a briefing."""

tokens = re.findall('\w+',text)

words = []

for word in tokens:
    words.append(word.lower())

nlp_words = nltk.FreqDist(words)
nlp_words.plot(20)


di = dict()
for w in words:
    if w in di:
        di[w] = di[w] + 1
    else:
        di[w] = 1
        
from nltk.corpus import gutenberg

raw = gutenberg.raw()
corpus = gutenberg.words()

#########
from nltk.corpus import PlaintextCorpusReader

#defining the Corpus Directory
dirname_politics = "E:\\NLP\Datsets\\mini_newsgroups\\talk.politics.misc"
#Reading the data with Corpus
politics_corpus  = PlaintextCorpusReader(dirname_politics,'.*')




