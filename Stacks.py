# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:12:13 2018

@author: giris
"""
from sys import maxsize

def createstack():
    stack = []
    return stack

#Stack is empty
    
def isEmpty(stack):
    return len(stack) == 0

#Function to add item on stack.It increases size by 1
def push(stack,item):
    stack.append(item)
    print(item + "Item pushed into stack")

#Fucntion to remove item from the stack. It decreases size by 1
def pop(stack):
    if (isEmpty(stack)):
        return str(-maxsize - 1) #returns minus infinite
    return stack.pop()

#Driver program to test the algorithm

stack = createstack()
push(stack,str(10))
push(stack,str(20))
push(stack,str(30))
print(pop(stack) + "popped from stack")
print(pop(stack) + "popped from stack")
print(pop(stack) + "popped from stack")
print(pop(stack) + "popped from stack")