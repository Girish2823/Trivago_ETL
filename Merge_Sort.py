# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 01:04:28 2018

@author: giris
"""

def merge_sort(A):
    merge_sort2(A,0,len(A)-1)

def merge_sort2(A,first,last):
    if first < last:
        middle = (first+last)//2
        merge_sort2(A,first,middle)
        merge_sort2(A,middle+1,last)
        merge(A,first,middle,last)

def merge(A,first,middle,last):
    L = A[first:middle]
    R = A[middle:last+1]
    L.append(99999999)
    R.append(99999999)
    i = j = 0
    for k in range (first,last+1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

A = [19,88,65,41,32,5,4,0,11]
