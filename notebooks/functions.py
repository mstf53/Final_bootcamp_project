
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

def build_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print('')
    print(f"Accuracy of train set: {round(accuracy_score(y_train, y_train_pred), 3)}")
    print('')
    print("Classification report of train set:") 
    print(classification_report(y_train, y_train_pred))
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp1 = ConfusionMatrixDisplay(cm_train, display_labels=model.classes_)
    disp1.plot()
    plt.show()
    
    print('')
    print(f"Accuracy of test set: {round(accuracy_score(y_test, y_test_pred), 3)}")
    print('')
    print("Classification report of test set:") 
    print(classification_report(y_test, y_test_pred))
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp2 = ConfusionMatrixDisplay(cm_test, display_labels=model.classes_)
    disp2.plot()
    plt.show()


def find_k_value(X_train, X_test, y_train, y_test):
    acc = []

    for i in range(1,40):
        neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
        acc.append(neigh.score(X_test, y_test))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),acc,color = 'blue', marker='o',markerfacecolor='red', markersize=7)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))