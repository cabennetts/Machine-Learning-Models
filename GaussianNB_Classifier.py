import sys
import scipy
import numpy as np
import pandas as pd
import sklearn 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# loads iris dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./iris.csv', names=names)

array = dataset.values # turn numpy dataset into array of values

split1a = array[0:25,:] # first 25 of iris-setosa
split2a = array[25:50,:] # second 25 of iris-setosa
split1b = array[50:75,:] # first 25 of iris-versicolor
split2b = array[75:100,:] # second 25 of iris-versicolor
split1c = array[100:125,:] # first 25 of iris-virginica
split2c = array[125:150,:] # second 25 of iris-virginica

a = np.concatenate([split1a, split1b])
b = np.concatenate([split2a, split2b])

firstHalf = np.concatenate([a, split1c]) 
secondHalf = np.concatenate([b, split2c])

X1 = firstHalf[:,0:4]
y1 = firstHalf[:,4]

X2 = secondHalf[:,0:4]
y2 = secondHalf[:,4]

# classifies the iris data set using the python built-in Naive Bayesian classifier, GuassianNB
model = GaussianNB()
model.fit(X1, y1)
pred1 = model.predict(X2)

#predicted = numpy.concatenate([pred1, pred2])
print("Accuracy Score: ",accuracy_score(y1, pred1)) # accuracy
# print out the confusion matrix
print ("Confusion Matrix:\n",confusion_matrix(y1, pred1)) # confusion matrix
# prints out the P, R, and F1 score for each of the 3 varieties of iris
print("Classification Report: \n",classification_report(y1, pred1)) # P, R, & F