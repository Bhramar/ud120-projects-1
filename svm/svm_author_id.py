#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print("BEFORE----")
print("len(features_train)")
print(len(features_train))
print("len(labels_train)")
print(len(labels_train))
#
#features_train = features_train[:len(features_train)//100] 
#labels_train = labels_train[:len(labels_train)//100] 
#
#print("AFTER------")
#print("len(features_train)")
#print(len(features_train))
#print("len(labels_train)")
#print(len(labels_train))

from sklearn.svm import SVC
clf = SVC(kernel="rbf",C=10000.0)
t0=time()
clf.fit(features_train,labels_train)
print("Time to train SVC(Kernel=RBF,C=10000.0):",round(time()-t0,4),"s")

t1=time()
pred=clf.predict(features_test)
print("Time to predict SVC(Kernel=RBF,C=10000.0):",round(time()-t1,4),"s")

print ("Prediction for element ")
print (pred)

count = (pred == 1).sum()
print(" cris count " )
print (count)

accuracy = clf.score(features_test,labels_test)
print ("accuracy")
print (accuracy)
#########################################################


