import glob
import os
import numpy as np
import pandas as pd
import heapq

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def create_classifiers():
    kernel = DotProduct() + WhiteKernel()
    return [
        KNeighborsClassifier(3),
        SVC(kernel='poly', gamma='scale', probability=True),
        SVC(gamma=2, C=1, probability=True),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        GaussianProcessClassifier(kernel=Matern(nu=2.5)),
        GaussianProcessClassifier(kernel=kernel),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, min_samples_leaf=5),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1, ),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]


def load_file(path: str):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace('class_0', 0)
    df = df.replace('class_1', 1)
    df = df.fillna(0)
    return df


def train_clasifiers(df: pd.DataFrame, classifiers, AUC: float = 0.0):
    kf = KFold(n_splits=10, shuffle=True)
    model = set()
    y = df['Class']
    X = df.drop(columns='Class')
    for classifier in classifiers:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_pred = classifier.fit(X_train, y_train).predict_proba(X_test)
            if hasattr(classifier, "predict_proba"):
                prob_pos = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = classifier.fit(X_train, y_train).decision_function(X_test)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            fpr, tpr, _ = roc_curve(y_test, prob_pos)
            roc_auc = auc(fpr, tpr)
            if roc_auc >= AUC:
                AUC = roc_auc
            # tpr = tp/(tp+fn)
            print("Area under the ROC curve : ", roc_auc)
            print('classifier: ', classifier)
    return model, AUC


def main():
    classifiers = create_classifiers()
    os.chdir("/Users/jesusllanogarcia/Desktop/Projecto/Clusters/CSV")
    AUC = {}
    for file in glob.glob("*.csv"):
        df = load_file(f'/Users/jesusllanogarcia/Desktop/Projecto/Clusters/CSV/{file}')
        model, auc = train_clasifiers(df, classifiers)
        print(auc)
        for classifier in model:
            print('Classifier: ', classifier)
        AUC[f'{file}'] = auc

    top_five = heapq.nlargest(5, AUC, key=AUC.get)

    print(top_five)
    for element in top_five:
        print(AUC[element])




if __name__ == "__main__":
    main()
