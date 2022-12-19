import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, cross_val_score

from sklearn.linear_model import LinearRegression # linear regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # polynomial regression
from sklearn.pipeline import Pipeline # for polynomial
from sklearn.naive_bayes import GaussianNB # Naive Bayesian
from sklearn.neighbors import KNeighborsClassifier #kNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# added to v2
from sklearn.svm import SVC # svm
from sklearn import tree # decision tree 
from sklearn.ensemble import RandomForestClassifier # random forest
from sklearn.ensemble import ExtraTreesClassifier # extra trees
from sklearn.neural_network import MLPClassifier # neural network 

# loads the iris data set
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./iris.csv', names=names)
array = dataset.values # turn numpy dataset into array of values

X = array[:,0:4] # contains flower features
y = array[:,4] # contains flower names

Parr3 = preprocessing.LabelEncoder()
PR3 = PolynomialFeatures(degree=6)
PR3.fit_transform(X)
Parr3.fit(y)
Parr3.transform(y)
d3X_train, d3X_test, d3y_train, d3y_test = train_test_split(X, Parr3.transform(y), test_size=0.50, random_state=1)

# convert class from strings to integers for training regression models
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

# Split data into 2 folds for training and test
X_train, X_test, y_train, y_test = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

# Polynomial of degree 3 regression
def Deg3LinReg_Test(x1, x2, y1, y2) :
  LR = LinearRegression().fit(x1, y1)
  pred1 = LR.predict(x2).round() # first fold testing
  LR.fit(x2, y2) # second fold training
  pred2 = LR.predict(x1).round() # second fold testing


  actual = np.concatenate([y2, y1])
  predicted = np.concatenate([pred1, pred2])
  predicted = predicted.round()
    
  print("Polynomial of degree 3 regression: \n")   
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
 
Deg3LinReg_Test(d3X_train, d3X_test, d3y_train, d3y_test) 