import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import KFold
from sklearn.metrics import SCORERS, roc_curve, auc
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def create_classifiers():
    return [
        KNeighborsClassifier(3),
        SVC(kernel='poly', gamma='scale', probability=True),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, min_samples_leaf=5),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        GaussianNB(),
        DecisionTreeClassifier(max_depth = 5, min_samples_split= 5), MLPClassifier(alpha=1, max_iter=5000)]
        

def load_file(path: str):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace('class_0', 0)
    df = df.replace('class_1', 1)
    df = df.fillna(0)
    return df


def train_clasifiers(df: pd.DataFrame, classifiers, AUC: float = 0.0):
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    model = set()
    y = df['Class']
    X = df.drop(columns='Class')
    for classifier in classifiers:
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            print('X_train: ', X_train, 'X_test: ', X_test)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print('y_train: ', y_train, 'y_test: ', y_test)
            y_score = classifier.fit(X_train, y_train).predict(X_test)
            print('class: ', y_score)
            print('Real class: ', y_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            if roc_auc >= AUC:
                AUC = roc_auc
                model.add(classifier)
            # tpr = tp/(tp+fn)
            print("Area under the ROC curve : ", roc_auc)
    return model, AUC

def main():
    classifiers = create_classifiers()
    df = load_file('/Users/jesusllanogarcia/Desktop/Projecto/Clusters/CSV/Cluster-76.csv')
    model, auc = train_clasifiers(df, classifiers)
    for classifier in model:
        print('Classifier: ', classifier)


if __name__ == "main":
    main()
