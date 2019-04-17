# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:06:18 2018

@author: giris
"""

#K-Fold Cross Validation
#Every time we build a model there are two types of parameters, one that we choose and the other that are learned by the model
#and are changed by the model after the model has run.
#For example - kernel parameter in Kernerl SVM - these are known as Hyper Parameters

#So, far we have broken down a dataset into test and training set, which is a correct way of evaluating model performance.
#But, not the best way, because we could have the variance problem. The Variance Problem can be explained,by the fact that when
#we get the accuracy on the test set. Well, if we run the model again and test again it 's performance on another test set.
#We,can get a very different accuracy. So, judging the accuracy on only test set is not relevant.

#So, we fix it by using the K-Fold Technique.

#Break down the training set to n folds and train the model on n-1 folds, and we test it on the last fold. Since, with n folds
#we can make n different combinations of n-1 combinations to train and 1 to test on.

#This will make a better idea of the model performance, because we can now take up the accuracies up to ten evaluations and also
#compute standard deviations.

#This will also give us in which of the 4 categories the our model will belong to :-
# 1. High Bias Low Variance 2.High Bias High Variance 3.Low Bias Low Variance 4.Low Bias High Variance

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('E:\Machine Learning\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 17 - Kernel SVM\Kernel_SVM\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
classifier.fit(X_train,y_train)

#Predicting thet test set result
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
#This will return 10 accuracies for each one of the 10 combinations that will be created through k-Fold cross validation.
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10)
accuracies.mean()

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()