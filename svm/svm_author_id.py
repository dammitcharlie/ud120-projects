#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
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
from sklearn.svm import SVC
clf = SVC(C=10000.0,kernel='rbf')

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
clf.fit(features_train, labels_train)

predicted_labels = clf.predict(features_test)

print predicted_labels[10], predicted_labels[26], predicted_labels[50]

#accuracy stuff
number_right = 0
for i in range(len(predicted_labels)):
    number_right = number_right + (predicted_labels[i]==labels_test[i])

accuracy = number_right/float(len(labels_test))
print accuracy


#how many test cases are predicted to be Chris, ie 1?
print sum(predicted_labels)



#########################################################


