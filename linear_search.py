# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:08:21 2018

@author: giris
"""

def linearsearch(list,tv):
    
    for i in range(0,len(list)):
        if list[i] == tv:
            return i #Function stops
    
    return -1
    



list = [2,6,5,2,4,7,3]
loc = linearsearch(list,77)

print("End Program")
print(loc)
    

    
