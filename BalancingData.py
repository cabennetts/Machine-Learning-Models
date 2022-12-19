import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
import imblearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.neural_network import MLPClassifier # neural network 
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks
# loads the iris data set
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./imbalanced_iris.csv', names=names)
array = dataset.values # turn numpy dataset into array of values
X = array[:,0:4] # contains flower features
y = array[:,4] # contains flower names
# convert class from strings to integers for training regression models
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

# Split data into 2 folds for training and test
X_train, X_test, y_train, y_test = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

def p1(x1, x2, y1, y2):
    print(" -=-=-=-= Part 1 =-=-=-=-")
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    predicted = predicted.round()

    cm = confusion_matrix(actual, predicted)
    """ Confusion Matrix Transposed from Lecture Slides
                                        Predicted
                            Setosa | Versicolor | virginica
                           |-------+------------+------------|
                    Setosa |   40  |     0      |     0      |    0
    True/                  |-------+------------+------------|
    Actuals     Versicolor |   0   |     27     |     3      |    1
                           |-------+------------+------------|
                 virginica |   0   |      0     |     50     |    2
                           -----------------------------------
                                0        1           2
    
    """
    num_classes = 3
    setosa_precision = cm[0][0]/(cm[0][0] + cm[0][1] + cm[0][2]) 
    setosa_recall =  cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])
    setosa_min = min(setosa_precision, setosa_recall)
    vcolor_precision = cm[1][1] / (cm[1][0] + cm[1][1] + cm[1][2])
    vcolor_recall = cm[1][1] / (cm[0][1] + cm[1][1] + cm[2][1])
    vcolor_min = min(vcolor_precision, vcolor_recall)
    v_precision = cm[2][2] / (cm[2][0] + cm[2][1] + cm[2][2])
    v_recall = cm[2][2] / (cm[0][2] + cm[1][2] + cm[2][2])
    v_min = min(v_precision, v_recall)
    cba = ((setosa_min + vcolor_min + v_min) / num_classes).round(3)

    # TN/(TN+FP)
    setosa_tn = cm[1][0] + cm[2][0] + cm[1][2] + cm[2][2]
    setosa_fp = cm[1][1] + cm[2][1]
    setosa_specificity = setosa_tn / (setosa_tn + setosa_fp)
    setosa = (setosa_recall + setosa_specificity) / 2

    vcolor_tn = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
    vcolor_fp = cm[0][1] + cm[2][1]
    vcolor_specificity = vcolor_tn / (vcolor_tn + vcolor_fp)
    vcolor = (vcolor_recall + vcolor_specificity) / 2
    v_tn = cm[0][0] + cm[0][2] + cm[1][0] + cm[1][2]
    v_fp = cm[0][1] + cm[1][1]
    v_specificity = v_tn / (v_tn + v_fp)
    virginica = (v_recall + v_specificity) / 2

    balanced_acc = ((setosa + vcolor + virginica) / num_classes).round(3)
    # calc_ba = 
    # print confusion matrix and accuracy score using imbalanced iris dataset
    print("Accuracy Score: ",accuracy_score(actual, predicted)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix
    # print and calculated class balanced accuracy 
    print("Class balanced accuracy: ", cba)
    # print calculated balanced accuracy 
    print("Calculated Balanced Accuract: ", balanced_acc)
    # print balanced accuracy score using balanced_accuracy_score
    bas = balanced_accuracy_score(actual, predicted).round(3)
    print("Balanced_accuracy_score(): ", bas)


def overSampling(X, y):
    print(" -=-=-=-= Part 2 =-=-=-=-")
    print("RandomOverSampling")
    # balance with random oversampling
    # X_train, X_test, y_train, y_test = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)
    ros = RandomOverSampler(random_state=0)
    randomX, randomY = ros.fit_resample(X, y)
    x1, x2, y1, y2 = train_test_split(randomX, randomY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix

    # balance with SMOTE
    print("SMOTE")
    smoteX, smoteY = SMOTE().fit_resample(X, y)
    x1, x2, y1, y2 = train_test_split(smoteX, smoteY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix
    

    # balance with ADASYN 
    print("ADASYN")
    # use sampling_strategy='minority'
    ada = ADASYN(random_state=0, sampling_strategy='minority')
    adaX, adaY = ada.fit_resample(X, y)
    
    x1, x2, y1, y2 = train_test_split(adaX, adaY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix
    

def underSampling(X, y):
    print(" -=-=-=-= Part 3 =-=-=-=-")
    # use imbalanced-learn toolbox

    # balance with random undersampling
    print("RandomUnderSampling")
    rus = RandomUnderSampler(random_state=0)
    randomX, randomY = rus.fit_resample(X, y)
    x1, x2, y1, y2 = train_test_split(randomX, randomY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix

    # balance with cluster undersampling 
    print("Cluster Undersampling")
    cluster = ClusterCentroids(random_state=0)
    clusterX, clusterY = cluster.fit_resample(X, y)
    x1, x2, y1, y2 = train_test_split(clusterX, clusterY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix

    # balance with tomek links
    print("Tomek links")
    tomek = TomekLinks()
    tomekX, tomekY = tomek.fit_resample(X, y)
    x1, x2, y1, y2 = train_test_split(tomekX, tomekY, test_size=0.50, random_state=1)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x1,y1)
    pred1 = clf.predict(x2)
    clf = clf.fit(x2, y2)
    pred2 = clf.predict(x1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred1, pred2])
    # predicted = predicted.round()
    cm = confusion_matrix(actual, predicted)
    # print out confusion matrix and accuracy score
    print("Accuracy Score: ",accuracy_score(actual, predicted).round(3)) # accuracy
    print("Confusion Matrix:\n",cm) # confusion matrix


p1(X_train, X_test, y_train, y_test)
overSampling(X, y)
underSampling(X, y)