from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import get_config


def create_classifiers() -> List:
    return [
        KNeighborsClassifier(3),
        SVC(kernel='poly', gamma='scale', probability=True),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, min_samples_leaf=5),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5, min_samples_split=5), MLPClassifier(alpha=1, max_iter=5000)]


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace('class_0', 0)
    df = df.replace('class_1', 1)
    df = df.fillna(0)
    return df


def get_best_classifier(df: pd.DataFrame, classifiers: List) -> Tuple[Any, float]:
    seed = get_config("INIT", "seed")
    if seed.isspace():
        seed = 1
    kf = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    best_auc = 0.0
    best_classifier = None
    original_class = df['class']
    df = df.drop(columns='class')
    for classifier in classifiers:
        for train_index, test_index in kf.split(df):
            training_set, test_set = df.iloc[train_index], df.iloc[test_index]
            training_class, test_class = original_class.iloc[train_index], original_class.iloc[test_index]
            predicted = classifier.fit(training_set, training_class).predict(test_set)
            fpr, tpr, _ = metrics.roc_curve(test_class, predicted)
            auc = metrics.auc(fpr, tpr)
            if auc > best_auc:
                best_auc = auc
                best_classifier = classifier
    return best_classifier, best_auc


def obtain_best_classifier_in_folder(directory: Path) -> List[Tuple[Any, float, Path]]:
    files = [x for x in directory.iterdir() if x.suffix == ".csv"]
    classifiers = create_classifiers()
    result = []
    for file in files:
        df = load_file(file)
        df = clean_dataset(df)
        classifier, auc = get_best_classifier(df, classifiers)
        result.append((classifier, auc, file))
    return result


def main():
    directory = Path("Data/Partitions").resolve()
    results = obtain_best_classifier_in_folder(directory)
    for classifier, auc, file in results:
        print(f"File {file.name} best classifier is {type(classifier).__name__} with auc {auc}")


if __name__ == "__main__":
    main()
