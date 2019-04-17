# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:31:14 2018

@author: giris
"""

import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

text = """The Senate on Wednesday delivered a historic rebuke of Saudi Arabia and President Trump’s handling of the fallout over journalist Jamal Khashoggi killing last month,
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


#The '\w' is a special character that will match any alphanumeric A-Z,a-z,0-9 along with underscores.
#The + tells you that the previous character in the regex can appear as many times as you want that you are trying to match.
tokens = re.findall('\w+',text)
tokens[0:]

#Now, let us convert the uppercases to lower cases
words = []

for word in tokens:
    words.append(word.lower())

words[0:]

#Removing Stop Words
sw = nltk.corpus.stopwords.words('english')

words_ns = []

for word in words:
    if word not in sw:
        words_ns.append(word)

#Creating a Dictionary to maintain the Count of Words
di = dict()
for w in words_ns:
    if w in di:
        di[w] = di[w] + 1
    else:
        di[w] = 1

print(di)

#Figures inline and set Visualizations style
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('darkgrid')
#Count of Words using the FreqDist Function
nlp_words = nltk.FreqDist(words_ns)
nlp_words.plot(20)



file = "E:\\NLP\\Datsets\\WHO_Report.txt"
handle = open(file)
d = dict()

for line in handle:
    line = line.rstrip()
    #print(line)
    wds = line.split()
    #print(wds)
    for w in wds:
        if w in d:
            d[w]=d[w] + 1
        else:
            d[w] = 1
print(d)

