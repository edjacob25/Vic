from configparser import ConfigParser
from datetime import datetime

from dateutil.relativedelta import relativedelta

config = None


def time_human_readable(start: float, end: float) -> str:
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
