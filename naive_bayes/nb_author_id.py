#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB

t0=time()

classifier = GaussianNB()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1=time()
predicted_labels=classifier.predict(features_test)
print "prediction:", round(time()-t1, 3), "s"

t1=time()

number_right = 0
for i in range(len(predicted_labels)):
    number_right = number_right + (predicted_labels[i]==labels_test[i])

accuracy = number_right/float(len(labels_test))
print accuracy
#########################################################


