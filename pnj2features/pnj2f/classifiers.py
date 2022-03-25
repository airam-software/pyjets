# Classification functions

import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def holdoutstraintest(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    start = time.time()
    clf.fit(X_train, y_train)  # train the model
    end = time.time()
    traintime = end - start

    start = time.time()
    predicted = clf.predict(X_test)
    end = time.time()
    classtime = end - start

    acc = accuracy_score(y_test, predicted)

    return traintime, classtime, acc, clf


def trainfull(X, y, clf):
    # Train with full expert dataset

    start = time.time()
    clf.fit(X, y)  # train the model
    end = time.time()
    traintime = end - start

    start = time.time()
    predicted = clf.predict(X)
    end = time.time()
    classtime = end - start

    y_final = predicted

    return traintime, classtime, y_final, clf


def leaveoneoutcv(x1, x2, y, month_vec, clf):
    acc_vec = []
    traintime_vec = []
    classtime_vec = []
    for idx, item in enumerate(month_vec):
        print(item)
        traintime, classtime, acc, _ = holdoutstraintest(x1, x2, y, np.delete(month_vec, idx), clf)
        acc_vec.append(acc)
        traintime_vec.append(traintime)
        classtime_vec.append(classtime)

    return np.average(traintime), np.average(classtime), np.average(acc_vec), clf


def classifyfull2(x1, x2, clf, nmonths):
    predicted = []
    for i in np.arange(nmonths):
        print('month' + str(i))
        X = np.vstack((x1[i, :, :].flatten(),
                       x2[i, :, :].flatten())).T.tolist()

        X = np.array(X)

        start = time.time()
        predicted.append(clf.predict(X))
        end = time.time()
        classtime = end - start

    y_final = predicted

    return y_final, classtime


def classify_composite2(x1, x2, clf):
    X = np.vstack((x1.flatten(),
                   x2.flatten())).T.tolist()
    X = np.array(X)

    start = time.time()
    predicted = clf.predict(X)
    end = time.time()

    return predicted, end - start
