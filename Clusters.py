from scipy.io import arff
import numpy as np
import pandas as pd
import random
import re


def create_r(seed=30):
    lists = []

    for i in range(10, 200, 10):
        print(i)

    for i in range(100-25, 100+26):
        lists.append(i)
    return lists


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


def main():
    data = arff.loadarff('dataset.NoString.MediumSize.arff')
    df = pd.DataFrame(data[0])
    df = df.drop(columns=[''])

    lists = create_r(30)

    print(lists)

    for r in lists:
        df1 = df.drop(columns=['rank_1'])
        class_series = pd.Series(['class_0' if i <= r else 'class_1' for i in range(200)])
        df1['Class'] = class_series
        df1.to_csv('/Users/jesusllanogarcia/Desktop/Projecto/Clusters/Cluster-{}.csv'.format(r))
        write_arff_file(df1, filename='/Users/jesusllanogarcia/Desktop/Projecto/Clusters/Cluster-{}.arff'.format(r))


if __name__ == '__main__':
    main()
    # fix_csv_names()
