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

# For Deg2 lin reg
Parr2 = preprocessing.LabelEncoder()
PR2 = PolynomialFeatures(degree=2)
PR2.fit_transform(X)
Parr2.fit(y)
Parr2.transform(y)
d2X_train, d2X_test, d2y_train, d2y_test = train_test_split(X, Parr2.transform(y), test_size=0.50, random_state=1)

# For De3 lin reg
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
  print ("#########################\n")
  print("Linear regression")
  print(accuracy_score(actual, predicted)) # accuracy
  print(confusion_matrix(actual, predicted)) # confusion matrix
  print ("#########################\n")
############################################################

#2  # Polynomial of degree 2 regression 
def Deg2LinReg_Test(x1, x2, y1, y2) :
  LR = LinearRegression().fit(x1, y1)
  pred1 = LR.predict(x2).round() # first fold testing
  LR.fit(x2, y2) # second fold training
  pred2 = LR.predict(x1).round() # second fold testing


  actual = np.concatenate([y2, y1])
  predicted = np.concatenate([pred1, pred2])
  predicted = predicted.round()
  
  print("Polynomial of degree 2 regression: \n")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
  print("#########################\n")
############################################################

#3  # Polynomial of degree 3 regression
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
  print("#########################\n")
############################################################

#4  # Naive Bayesian
def NaiveBayes_Test(x1, x2, y1, y2):

  model = GaussianNB()
  model.fit(x1, y1) # first fold training
  pred1 = model.predict(x2) # first fold testing
  model.fit(x2, y2) # second fold training
  pred2 = model.predict(x1) # second fold testing

  # Evaluate Model
  actual = np.concatenate([y2, y1]) # actual classes
  predicted = np.concatenate([pred1,pred2]) # predicted classes
  predicted = predicted.round()

  print("Naive Bayesian:")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
  print("#########################\n")
############################################################

#5  # KNeighbors Classifier
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
  print("#########################\n")
############################################################

#6  # Linear Discriminant Analysis
def LinDiscriminant_Test(x1, x2, y1, y2):
  # #Fit the LDA model
  LDA = LinearDiscriminantAnalysis()
  LDA.fit(x1, y1)
  pred1 = LDA.predict(x2)
  LDA.fit(x2, y2)
  pred2 = LDA.predict(x1)

  # Evaluate Model
  actual = np.concatenate([y2, y1]) # actual classes
  predicted = np.concatenate([pred1,pred2]) # predicted classes
  predicted = predicted.round()

  print("Linear Discriminant Analysis:")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
  print("#########################\n")
############################################################

#7  # Quadratic Discriminant Analysis
def QuadDiscriminant_Test(x1, x2, y1, y2):
  # x = features
  # y = classes (flowers)

  QDA = QuadraticDiscriminantAnalysis()
  QDA.fit(x1,y1)
  pred1 = QDA.predict(x2)
  QDA.fit(x2,y2)
  pred2 = QDA.predict(x1)

  # Evaluate Model
  actual = np.concatenate([y2, y1]) # actual classes
  predicted = np.concatenate([pred1,pred2]) # predicted classes
  predicted = predicted.round()

  print("Quadratic Discriminant Analysis:")
  print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
  print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
  print("#########################\n")
############################################################

#8  # SVM
def Svm(x1, x2, y1, y2):
    s = SVC(kernel = 'linear', random_state=0)
    s.fit(x1, y1)
    pred1 = s.predict(x2)
    s.fit(x2, y2)
    pred2 = s.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()
    
    print("Support Vectored Machine:")
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
    print("#########################\n")
############################################################

#9  # Decision Tree
def DecTree(x1, x2, y1, y2):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x1, y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()
    
    print("Decision Trees:")
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
    print("#########################\n")
############################################################

#10 # Random Forest
def RandomForest(x1, x2, y1, y2):
    clf = RandomForestClassifier()
    clf = clf.fit(x1, y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()
    
    print("Random Forest:")
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
    print("#########################\n")

############################################################

#11 # ExtraTrees
def ETrees(x1, x2, y1, y2):
    clf = ExtraTreesClassifier()
    clf = clf.fit(x1, y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()
    
    print("Extra Trees:")
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
    print("#########################\n")

############################################################

#12 # Neural Network
def NN(x1, x2, y1, y2):
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()

    print("Neural Network:")
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",confusion_matrix(actual, predicted)) # confusion matrix
    print("#########################\n")

############################################################

LinReg_Test(X_train, X_test, y_train, y_test) 
Deg2LinReg_Test(d2X_train, d2X_test, d2y_train, d2y_test) 
Deg3LinReg_Test(d3X_train, d3X_test, d3y_train, d3y_test) 
NaiveBayes_Test(X_train, X_test, y_train, y_test)
KNeighbors_Test(X_train, X_test, y_train, y_test) 
LinDiscriminant_Test(X_train, X_test, y_train, y_test) 
QuadDiscriminant_Test(X_train, X_test, y_train, y_test)
Svm(X_train, X_test, y_train, y_test)
DecTree(X_train, X_test, y_train, y_test)
RandomForest(X_train, X_test, y_train, y_test)
ETrees(X_train, X_test, y_train, y_test)
NN(X_train, X_test, y_train, y_test)
