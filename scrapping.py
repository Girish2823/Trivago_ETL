# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:38:13 2018

@author: girish
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd

# read the data from a URL
url = requests.get("https://cs.gmu.edu/~hfoxwell/580books.html")

# parse the URL using Beauriful Soup
soup = BeautifulSoup(url.content, 'html.parser')

# find all HTML <li> entries
listitems = soup.find_all('li')

# open an output file for Writing the csv data
filename = "listofbooks.csv"
f = open(filename, "w")

# write the data item header
headers = "title, author, publisher, release\n"
f.write(headers)

# find and clean up the list items (note the commas)
for entry in listitems:
    title = entry.a.booktitle.text
    title = title.replace(",", "|")
    author = entry.author.text
    author = author.replace(",", "|")
    publisher = entry.publisher.text
    publisher = publisher.replace(",", "|")
    release = entry.release.text
    release = release.replace(",", "|")
# write the data list items, don't forget the newline    
    f.write(title + "," + author + "," + publisher + "," + release + "\n")
# close the file (finalize the writes)
f.close()

# Alternatively, if the data is in a well-formated table:
# find the table in the URL
for record in soup.findAll('tr'):
# start building the record with an empty string
    tbltxt = ""
# find all the table data strings, add to tbltxt
    for data in record.findAll('td'):
        tbltxt = tbltxt + data.text + ","
# display the record; drop the trailing comma
    print(tbltxt)
    print(tbltxt[0:-1])
    print()