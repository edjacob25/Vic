import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.io import arff


def create_index_partitions() -> List[int]:
    partitions = []
    for i in range(10, 200, 10):
        partitions.append(i)

    for i in range(75, 126):
        partitions.append(i)
    return partitions


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


def create_partitions(dataset_path: Path):
    data, a = arff.loadarff(dataset_path)
    df = pd.DataFrame(data)
    partitions = create_index_partitions()
    base_path = Path("Data/Partitions")
    base_path = base_path.resolve()
    if not base_path.exists():
        base_path.mkdir()

    for r in partitions:
        df = df.drop(columns=['rank_1'])
        class_series = pd.Series(['class_0' if i <= r else 'class_1' for i in range(200)])
        df['class'] = class_series
        path = base_path / f"partition_{r}"
        df.to_csv(path.with_suffix(".csv"))
        write_arff_file(df, filename=path.with_suffix(".arff"))


def main():
    dataset_path = Path("Data/initial_dataset.arff")
    create_partitions(dataset_path)


if __name__ == '__main__':
    main()
