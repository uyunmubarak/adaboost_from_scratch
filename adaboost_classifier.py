#import library
import numpy as np
import pandas as pd

from ml_from_scratch.ensemble import AdaboostClassifier
from ml_from_scratch.ensemble import DecisionTreeMaxDepth1
from ml_from_scratch.metrics import accuracy_score

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    #Create an AdaboostClassifier object
    clfs = AdaboostClassifier(estimator=DecisionTreeMaxDepth1())

    #Train the model using the training data
    clfs.fit(X_train, y_train)

    #Perform prediction on test data
    y_pred = clfs.predict(X_test)
    print("Prediksi Data :", y_pred)

    #Calculating prediction accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy :", acc)

