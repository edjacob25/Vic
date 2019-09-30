import re
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

config = None


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


def format_time_difference(start: float, end: float) -> str:
    attrs = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    delta = relativedelta(datetime.fromtimestamp(end), datetime.fromtimestamp(start))
    spaces = ['%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1]) for attr in attrs if
              getattr(delta, attr)]
    return ", ".join(spaces)


def get_config(section: str, config_name: str) -> str:
    global config
    if config is None:
        config = ConfigParser()
        config.read("config.ini")
    return config[section][config_name]


def write_arff_file(dataset: pd.DataFrame, filename="dataset.arff", name="Universities"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"@RELATION {name}\n\n")
        max_len = len(max(dataset.columns, key=len))
        for header in dataset.columns:
            if dataset[header].dtype == np.float64 or dataset[header].dtype == np.int64:
                column_type = "NUMERIC"
            else:
                column_type = "{'class_0','class_1'}"

            file.write(f"@ATTRIBUTE {header.ljust(max_len)} {column_type}\n")
        file.write("\n@DATA\n")

        for _, column in dataset.iteritems():
            if column.dtype == np.object:
                pattern = re.compile(r"^(.*)$")
                dataset[column.name] = column.str.replace(pattern, r'"\1"')

        for _, row in dataset.iterrows():
            items = [str(x) for x in row]
            items = [x if x != "nan" else "?" for x in items]
            file.write(f"{', '.join(items)}\n")
