import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

#1  # Linear Regression
def LinReg_Test(x1, x2, y1, y2) :
  # Perform 2-fold cross-validation
  LR = LinearRegression().fit(x1, y1)
  pred1 = LR.predict(x2).round() # first fold testing
  LR.fit(x2, y2) # second fold training
  pred2 = LR.predict(x1).round() # second fold testing

  # Evaluate Model
  actual = np.concatenate([y2, y1]) # actual classes
  predicted = np.concatenate([pred1,pred2]) # predicted classes
  predicted = predicted.round()
  print("Linear regression")
  print(accuracy_score(actual, predicted)) # accuracy
  print(confusion_matrix(actual, predicted)) # confusion matrix

LinReg_Test(X_train, X_test, y_train, y_test)