import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Models
from sklearn import svm 
from sklearn import tree
from sklearn import linear_model

#Import CSV and get labels and data
train_set = pd.read_csv('./train.csv')
label_set = train_set['label']
train_set = train_set.drop('label', axis = 1)

def printIndex(sample):
    sample=sample.values.reshape((28,28))
    plt.imshow(sample,cmap='gray')
    plt.show()

data_set_sizes = [125, 625, 1250, 6250, 12500, 25000]
svm_model = svm.SVC(gamma='scale', decision_function_shape = 'ovo')
tree_model = tree.DecisionTreeClassifier()
lr_model = linear_model.LogisticRegression()
svm_scores = []
tree_scores = []
lr_scores = []

# Create data for multiple sizes
for size in data_set_sizes:
    print("Testing Models at Size:  ", size)
    # Get data by size and Split
    set_X = train_set.iloc[0:size]
    set_Y = label_set.iloc[0:size]
    X_train, X_test, y_train, y_test = train_test_split(set_X, set_Y, test_size=0.2, random_state=21)
    # Fit all models
    svm_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    # Predict 
    svm_pred = svm_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    # Save scores
    svm_scores.append(accuracy_score(y_test, svm_pred))
    tree_scores.append(accuracy_score(y_test, tree_pred))
    lr_scores.append(accuracy_score(y_test, lr_pred))
    print("SVM at size: ", size)
    print(classification_report(y_test, svm_pred))
    print("Accuracy: ", accuracy_score(y_test, svm_pred))
    print("Tree at size: ", size)
    print(classification_report(y_test, tree_pred))
    print("Accuracy: ", accuracy_score(y_test, tree_pred))
    print("Logistic Regression at size: ", size)
    print(classification_report(y_test, lr_pred))
    print("Accuracy: ", accuracy_score(y_test, lr_pred))

# Test with full set

X_train, X_test, y_train, y_test = train_test_split(train_set, label_set, test_size=0.2, random_state=21)
# Fit all models
svm_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
# Predict 
svm_pred = svm_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
# Save scores
svm_scores.append(accuracy_score(svm_pred, y_test))
tree_scores.append(accuracy_score(tree_pred, y_test))
lr_scores.append(accuracy_score(lr_pred, y_test))
print("SVM at size: ", size)
print(classification_report(svm_pred, y_test))
print("Tree at size: ", size)
print(classification_report(tree_pred, y_test))
print("Logistic Regression at size: ", size)
print(classification_report(lr_pred, y_test))
