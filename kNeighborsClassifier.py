import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, cross_val_score

from sklearn.neighbors import KNeighborsClassifier #kNN

# loads the iris data set
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./iris.csv', names=names)
array = dataset.values # turn numpy dataset into array of values

X = array[:,0:4] # contains flower features
y = array[:,4] # contains flower names

# convert class from strings to integers for training regression models
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

# Split data into 2 folds for training and test
X_train, X_test, y_train, y_test = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

# KNeighbors Classifier
def KNeighbors_Test(x1, x2, y1, y2): 
  # x = features
  # y = classes (flowers)
  
  knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='euclidean')
  knn.fit(x1, y1)
  pred1 = knn.predict(x2)
  knn.fit(x2, y2)
  pred2 = knn.predict(x1)

  # Evaluate Model
  actual = np.concatenate([y2, y1]) # actual classes
  predicted = np.concatenate([pred1,pred2]) # predicted classes
  predicted = predicted.round()

  print("KNeighbors Classifier:")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix

KNeighbors_Test(X_train, X_test, y_train, y_test)