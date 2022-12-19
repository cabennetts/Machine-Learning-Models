import sys
import scipy
import numpy as np
import pandas as pd
import sklearn
from numpy.core.fromnumeric import size
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# for plotting 
def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()
# loads the iris data set
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('./iris.csv', names=names)
array = dataset.values # turn numpy dataset into array of values

X = array[:,0:4] # contains flower features
y = array[:,4] # contains flower names

correct = np.empty(shape=size(y))
for i in range(size(correct)):
    if y[i] == 'Iris-setosa':
        correct[i] = 0
    elif y[i] == 'Iris-versicolor':
        correct[i] = 1
    elif y[i] == 'Iris-virginica':
        correct[i] = 2

#################################################################
# takes 2nd derivate and finds the highest value
def find_elbow(arr):
    d = 0
    m = 0
    for i in range(2, size(arr)-1):
        apx = (arr[i+1] + arr[i-1] - 2 * arr[i])
        if d < apx:
            d = apx
            m = i
    return m + 1

def p1(X, y, correct):
    ### Part 1: k-Means Clustering ###
    print("Part 1: k-Means Clustering")
    print("-- k = elbow_k --")
    recons_err_arr = np.empty(shape=(20))
    for i in range(size(recons_err_arr)):
        kmeans = KMeans(n_clusters=i+1, random_state=0).fit(X)
        kmeans.predict(X)
        recons_err_arr[i]=kmeans.inertia_

    # plot_graph(recons_err_arr, 'Reconstruction Error')
    elbow_k = find_elbow(recons_err_arr)
    kmeans = KMeans(n_clusters=elbow_k, random_state=0).fit(X)
    kmeans.predict(X)

    labels = kmeans.labels_
    lbls_matched = np.empty_like(labels)
    for i in np.unique(labels):
        match = [np.sum((labels==i)*(correct==j)) for j in np.unique(correct)]
        lbls_matched[labels==i] = np.unique(correct)[np.argmax(match)]

    print("Confusion Matrix: ")
    print(confusion_matrix(correct, lbls_matched))
    print("Accuracy Score: ", accuracy_score(correct, lbls_matched))

    print("-- k=3 --")
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    kmeans.predict(X)

    labels = kmeans.labels_
    lbls_matched = np.empty_like(labels)
    for i in np.unique(labels):
        match = [np.sum((labels==i)*(correct==j)) for j in np.unique(correct)]
        lbls_matched[labels==i] = np.unique(correct)[np.argmax(match)]

    print("Confusion Matrix: ")
    print(confusion_matrix(correct, lbls_matched))
    print("Accuracy Score: ", accuracy_score(correct, lbls_matched))
##############################################################################

def p2(X, y, correct):
    ### Part 2: Gaussian Mixture Models (GMM) ###
    print("Part 2: Gaussian Mixture Models (GMM)")
    print ("-- AIC --")
    aic = np.empty(shape=(20,1))
    for i in range(1,21):
        gm = GaussianMixture(n_components=i, random_state=0, covariance_type='diag').fit(X)
        # gm.predict(X)
        aic[i-1] = gm.aic(X)
    # plot_graph(aic, 'aic')

    aic_elbow = 3
    gm_aic = GaussianMixture(n_components=aic_elbow, random_state=0, covariance_type='diag').fit(X)
    aic_predict = gm_aic.predict(X)
    aic_lbls_matched = np.empty_like(aic_predict)

    for k in np.unique(aic_predict):
            numsMatch = [np.sum((aic_predict == k) * (correct == t)) for t in np.unique(correct)]
            aic_lbls_matched[aic_predict == k] = np.unique(correct)[np.argmax(numsMatch)]

    print("Confusion Matrix: ")
    print(confusion_matrix(correct, aic_lbls_matched))
    print("Accuracy Score: ", accuracy_score(correct, aic_lbls_matched))

    print ("-- BIC --")
    bic = np.empty(shape=(20,1))
    for i in range(1,21):
        gm = GaussianMixture(n_components=i, random_state=0, covariance_type='diag').fit(X)
        # gm.predict(X)
        bic[i-1] = gm.bic(X)
    # plot_graph(bic, 'bic')

    bic_elbow = 4
    gm_bic = GaussianMixture(n_components=bic_elbow, random_state=0, covariance_type='diag').fit(X)
    bic_predict = gm_bic.predict(X)
    bic_lbls_matched = np.empty_like(bic_predict)

    # for k in np.unique(bic_predict):
    #         numsMatch = [np.sum((bic_predict == k) * (correct == t)) for t in np.unique(correct)]
    #         bic_lbls_matched[bic_predict == k] = np.unique(correct)[np.argmax(numsMatch)]

    print("Confusion Matrix: ")
    print(confusion_matrix(correct, bic_predict))
    print("Accuracy Score: Cannot calc accuracy score because the number of classes is not the same")

    print ("-- k=3 --")
    k = 3
    gm_aic = GaussianMixture(n_components=k, random_state=0, covariance_type='diag').fit(X)
    aic_predict = gm_aic.predict(X)
    aic_lbls_matched = np.empty_like(aic_predict)

    for k in np.unique(aic_predict):
            numsMatch = [np.sum((aic_predict == k) * (correct == t)) for t in np.unique(correct)]
            aic_lbls_matched[aic_predict == k] = np.unique(correct)[np.argmax(numsMatch)]

    print("Confusion Matrix: ")
    print(confusion_matrix(correct, aic_lbls_matched))
    print("Accuracy Score: ", accuracy_score(correct, aic_lbls_matched))
##############################################################################

p1(X, y, correct)
p2(X, y, correct)
