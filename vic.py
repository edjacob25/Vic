import math
import multiprocessing
import random
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sty import fg

from common import get_config, format_time_difference

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def create_classifiers() -> List:
    kernel = DotProduct() + WhiteKernel()
    return [KNeighborsClassifier(3),
            SVC(kernel='poly', gamma='scale', probability=True),
            SVC(gamma=2, C=1, probability=True),
            GaussianProcessClassifier(kernel=Matern(nu=2.5)),
            GaussianProcessClassifier(kernel=kernel),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, min_samples_leaf=5),
            RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1, ),
            GaussianNB(),
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', tol=0.0001),
            QuadraticDiscriminantAnalysis()]


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(str(path))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace('class_0', 0)
    df = df.replace('class_1', 1)
    df = df.fillna(0)
    return df


def calculate_auc(df: pd.DataFrame, classifier, k_fold: KFold, original_class: pd.DataFrame) -> Tuple[Any, float]:
    aucs = []
    for train_index, test_index in k_fold.split(df, original_class):
        training_set, test_set = df.iloc[train_index], df.iloc[test_index]
        training_class, test_class = original_class.iloc[train_index], original_class.iloc[test_index]
        if hasattr(classifier, "predict_proba"):
            predicted = classifier.fit(training_set, training_class).predict_proba(test_set)[:, 1]
        else:
            prob_pos = classifier.fit(training_set, training_class).decision_function(test_set)
            predicted = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fpr, tpr, _ = metrics.roc_curve(test_class, predicted)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
    return classifier, sum(aucs) / k_fold.n_splits


def get_best_classifier(df: pd.DataFrame, classifiers: List, seed: int, pool: multiprocessing.Pool) -> Tuple[Any, float]:
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    original_class = df['class']
    df = df.drop(columns='class')
    results = pool.starmap(calculate_auc, [(df, classifier, kf, original_class) for classifier in classifiers])
    results.sort(key=lambda x: x[1], reverse=True)

    return results[0]


def obtain_best_classifier_in_folder(directory: Path, seed: int) -> List[Tuple[Any, float, Path]]:
    procs = get_config("INIT", "procs")
    if procs.isspace() or not procs.isnumeric():
        procs = math.floor(multiprocessing.cpu_count() / 2)
    else:
        procs = int(procs)

    pool = multiprocessing.Pool(procs)
    files = [x for x in directory.iterdir() if x.suffix == ".csv"]
    classifiers = create_classifiers()
    result = []
    for file in files:
        df = load_file(file)
        df = clean_dataset(df)
        start = datetime.now()
        classifier, auc = get_best_classifier(df, classifiers, seed, pool)
        end = datetime.now()
        result.append((classifier, auc, file))
        print(
            f"Finished file {fg.blue}{file}{fg.rs}, took {format_time_difference(start.timestamp(), end.timestamp())}")
    pool.close()
    return result


def multiple_analysis(num_of_seeds):
    directory = Path("Data/Partitions").resolve()
    seeds = [random.randint(1, 10000000) for _ in range(num_of_seeds)]
    total_results = pd.DataFrame()
    for seed in seeds:
        results = obtain_best_classifier_in_folder(directory, seed)
        idx = [x[2].stem for x in results]
        data = [x[1] for x in results]
        series = pd.Series(name=f"{seed}", data=data, index=idx)
        total_results = total_results.append(series)
    print(total_results)
    total_results.to_csv("Data/vic_results.csv")
    print(total_results.mean())
    print(total_results.mean().sort_values())


def main():
    args = ArgumentParser()
    args.add_argument("-d", "--directory", required=True, help="File to be analyzed")
    args.add_argument("-s", "--seed", required=False, help="Seed which wants to be used, must be numeric", type=int)
    args.parse_args()

    directory = Path(args.directory).resolve()
    if not args.seed:
        seed = get_config("INIT", "seed")
        if seed.isspace() or not seed.isnumeric():
            seed = 1
        else:
            seed = int(seed)
    else:
        seed = args.seed

    start = datetime.now()
    results = obtain_best_classifier_in_folder(directory, seed)
    end = datetime.now()
    print(f"Analysis of all files took {format_time_difference(start.timestamp(), end.timestamp())}")
    for classifier, auc, file in results:
        print(f"File {file.name} best classifier is {type(classifier).__name__} with auc {auc}")

    rs = len(results)
    results = [x for x in results if not math.isnan(x[1])]
    print(f"Removed {rs - len(results)} that have nan")

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"{fg.blue}The best 5 splits are:{fg.rs}")
    for _, auc, file in results[:5]:
        print(f"- {fg.green}{file}{fg.rs} with auc {fg.green}{auc}{fg.rs}")


if __name__ == "__main__":
    main()
