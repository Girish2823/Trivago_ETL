# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 23:00:50 2018

@author: giris
"""

def bubblesort(mylist):
    for i in range(0,len(mylist)-1):
        for j in range(0,len(mylist)-1-i):
            if mylist[j] > mylist[j+1]:
                mylist[j],mylist[j+1] = mylist[j+1],mylist[j]
                
    return mylist

mylist = [17,9,0,78,9,-1,6]
print(bubblesort(mylist))
    