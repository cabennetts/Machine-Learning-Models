import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # polynomial regression
from sklearn.pipeline import Pipeline # for polynomial
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# loads the iris data set
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./iris.csv', names=names)
array = dataset.values # turn numpy dataset into array of values

X = array[:,0:4] # contains flower features
print(X)
y = array[:,4] # contains flower names
# convert class from strings to integers for training regression models
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

# Split data into 2 folds for training and test
X_train, X_test, y_train, y_test = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

# Standard_obj = StandardScaler()
# Standard_obj.fit(X_train)
# x_train_std = Standard_obj.transform(X_train)
# x_test_std = Standard_obj.transform(X_test)

# Polynomial of degree 2 regression 
def Deg2LinReg_Test(x1, x2, y1, y2) :
  # x = features
  # y = classes (flowers)
  array = preprocessing.LabelEncoder()
  PR2 = PolynomialFeatures(degree=2)
  x_poly = PR2.fit_transform(x1)
  array.fit(y1)
  array.transform(y1)

  # PR2 = PR2.fit_transform(x1)
  lin_reg2 = LinearRegression()
  lin_reg2.fit(PR2, y1)
  pred1 = lin_reg2.predict(x2)

  # fold 2
  # PR = PolynomialFeatures(degree=2)
  # pred1 = PR.predict(x2).round()
  # PR.fit(x2, y2)
  # pred2 = PR.predict(x1).round()

  actual = np.concatenate([y2, y1])
  predicted = np.concatenate([pred1, pred2])
  predicted = predicted.round()
  
  print("Polynomial of degree 2 regression: \n")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix

Deg2LinReg_Test(X_train, X_test, y_train, y_test) 