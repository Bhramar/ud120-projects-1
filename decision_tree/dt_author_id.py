#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

feat = len(features_train[0])
print("features",feat)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0=time()
clf = clf.fit(features_train,labels_train)
print("Traing time",round(time()-t0,4), "s")

acc = clf.score(features_test,labels_test)
print("Accuracy:", str(acc))

## Output :
    
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#features 379
#Traing time 4.6321 s
#Accuracy: 0.966439135381

#########################################################


