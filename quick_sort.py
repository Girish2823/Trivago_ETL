# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:43:47 2018

@author: giris
"""

def partition(arr,low,high):
    i = (low-1) #Index of smaller element
    pivot = arr[high] #pivot element
    
    for j in range(low,high):
        
        #If current element is smaller than the pivot element
        if arr[j] <= pivot:
            #Increment the index of smaller element
            i = i+1
            arr[i],arr[j] = arr[j],arr[i]
    arr[i+1],arr[high] = arr[high],arr[i+1]
    return (i+1)
    


def quicksort(arr,low,high):
    if ((high - low) > 0):
        #p is the partitioning index arr[p] is now at tight place
        p = partition (arr,low,high)
        
        #separately sort before partition and after partition
        quicksort(arr,low,p-1)
        quicksort(arr,p+1,high)

#Driver Code to check the algorithm
arr = [10,7,8,9,1,5]
n = len(arr)
quicksort(arr,0,n-1)

print("Sorted Array is : ")
for i in range(n):
    print(arr[i])
        
