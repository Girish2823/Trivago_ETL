# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:20:38 2018

@author: giris
"""


def binarysearch(data,target):
    low = 0
    high = len(data) - 1
    
    while low <= high:
        mid = (low + high)//2
        
        if target == data[mid]:
           return True
        elif target < data[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return False
        
         
   

data = [2,4,5,7,9,12,14,17,19,22,27,28,33,37]
target = 2

res = binarysearch(data,target)
print(res)