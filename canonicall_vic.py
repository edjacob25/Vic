import math
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

from common import load_file, clean_dataset, create_classifiers


def create_clusters(path: Path, num_of_clusters: int = 2) -> pd.DataFrame:
    df = load_file(path)
    df = clean_dataset(df)
    clusterer = KMeans(n_clusters=num_of_clusters).fit(df)
    res = df["assigned_cluster"] = clusterer.labels_
    return res


def split_dataset(df: pd.DataFrame, num_of_parts: int = 5) -> List[pd.DataFrame]:
    if df.shape[0] < num_of_parts:
        raise ValueError("Dataset must be bigger than the number of splits")
    return [df.iloc[i * num_of_parts: (i + 1) * num_of_parts - 1] for i in range(num_of_parts)]


def calculate_auc(classifier, training: pd.DataFrame, testing: pd.DataFrame) -> float:
    training_set, training_class = training.loc[: training.columns != "assigned_cluster"], \
                                   training.loc["assigned_cluster"]
    test_set, test_class = testing.loc[: testing.columns != "assigned_cluster"], testing.loc["assigned_cluster"]
    if hasattr(classifier, "predict_proba"):
        predicted = classifier.fit(training_set, training_class).predict_proba(test_set)[:, 1]
    else:
        prob_pos = classifier.fit(training_set, training_class).decision_function(test_set)
        predicted = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fpr, tpr, _ = metrics.roc_curve(test_class, predicted)
    auc = metrics.auc(fpr, tpr)
    return auc


def vic(path: Path, num_partitions=5):
    d = create_clusters(path)
    z = split_dataset(d, num_partitions)
    v = 0.0
    classifiers = create_classifiers()
    for classifier in classifiers:
        v_prime = 0.0
        for item in z:
            training = d.drop(item.index)
            auc = calculate_auc(classifier, training, item)
            v_prime += auc
        v = max(v, v_prime / num_partitions)
    return v


def vic_parallel(path: Path, num_partitions=5, procs:int=None):
    d = create_clusters(path)
    z = split_dataset(d, num_partitions)
    v = 0.0
    classifiers = create_classifiers()
    if procs is None:
        procs = math.floor(multiprocessing.cpu_count() / 2)
    if num_partitions < procs:
        procs = num_partitions
    pool = multiprocessing.Pool(procs)
    for classifier in classifiers:
        results = pool.starmap(calculate_auc, [(classifier, d.drop(item.index), item) for item in z])
        v_prime = sum(results)
        v = max(v, v_prime / num_partitions)
    pool.close()
    return v


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-f", "--file", help="CSV file which the initial dataset is located", required=True)
    args.add_argument("-s", "--splits", type=int, help="Num of splits that want to be used", default=5)
    args.add_argument("-p", "--parallel", action="store_true", help="Run the parallel version")
    args.add_argument("-c", "--cores", type=int, help="Num of cores to run the parallel version, if not specified, "
                                                      "it will take half of your current processors or the number of "
                                                      "splits, whichever is lower")
    args.parse_args()

    path = Path(args.file).resolve()

    if not path.exists():
        print("Required file does not exists")
        exit(1)

    if args.parallel:
        vic_parallel(path, args.splits, args.cores)
    else:
        vic(path, args.splits)


