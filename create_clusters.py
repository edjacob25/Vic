import re
from pathlib import Path
from typing import List

import pandas as pd
from scipy.io import arff

from common import write_arff_file


def create_index_partitions() -> List[int]:
    partitions = []
    for i in range(10, 200, 10):
        partitions.append(i)

    for i in range(75, 126):
        partitions.append(i)
    return partitions


def create_partitions(dataset_path: Path):
    data, a = arff.loadarff(dataset_path)
    df = pd.DataFrame(data)
    df = df.drop(columns=['rank_1'])
    partitions = create_index_partitions()
    base_path = Path("Data/Partitions")
    base_path = base_path.resolve()
    if not base_path.exists():
        base_path.mkdir()

    for r in partitions:
        class_series = pd.Series(['class_0' if i < r else 'class_1' for i in range(200)])
        df['class'] = class_series
        path = base_path / f"partition_{r}"
        df.to_csv(path.with_suffix(".csv"))
        write_arff_file(df, filename=path.with_suffix(".arff"))


def main():
    dataset_path = Path("Data/initial_dataset.arff")
    create_partitions(dataset_path)


if __name__ == '__main__':
    main()
