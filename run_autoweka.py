import math
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

from sty import fg

from common import get_config, format_time_difference


def run_on_file(dataset_path: Path, time_limit: int, java_mem: int = 8192, mem_limit: int = None, procs: int = None,
                folds: int = None, seed: str = None, metric: str = None) -> Tuple[str, str]:
    if os.name is not "posix":
        separator = ";"
    else:
        separator = ":"
    command = ["java", f"-Xmx{java_mem}m"]
    java_classpath = f"{get_config('ROUTES', 'weka_jar')}{separator}{get_config('ROUTES', 'autoweka_jar')}"
    command.append("-cp")
    command.append(java_classpath)
    command.append("weka.classifiers.meta.AutoWEKAClassifier")
    command.append("-t")
    command.append(str(dataset_path.resolve()))
    command.append("-timeLimit")
    command.append(str(time_limit))
    command.append("-parallelRuns")
    if not procs:
        procs = math.floor(multiprocessing.cpu_count() / 2)
    command.append(str(procs))
    if mem_limit:
        command.append("-memLimit")
        command.append(str(time_limit))

    if folds:
        command.append("-x")
        command.append(str(folds))

    if seed:
        command.append("-s")
        command.append(seed)

    if metric:
        command.append("-metric")
        command.append(metric)

    start = time.time()
    print("Command to run: ")
    print(f"{fg.blue}{' '.join(command)}{fg.rs}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.args)
    print(result.returncode)
    end = time.time()
    print(f"Took {fg.green}{format_time_difference(start, end)}{fg.rs} to run")
    return result.stdout.decode("utf-8"), result.stderr.decode("utf-8")


def main():
    data_dir = Path(".") / 'Data'

    initial_dir = data_dir / 'Initial'
    results_dir = data_dir / 'Results'
    if not data_dir.exists():
        data_dir.mkdir()
        initial_dir.mkdir()
        results_dir.mkdir()

    files = [x for x in initial_dir.iterdir() if x.is_file()]
    for data_file in files:
        weka_result, weka_error = run_on_file(data_file, time_limit=5, folds=5, seed=get_config("INIT", "seed"),
                                              metric="areaUnderROC")
        res_file = results_dir / data_file.with_suffix(".txt").name
        err_file = results_dir / data_file.with_suffix(".txt").name.replace(".t", "_error.t")
        with res_file.open("w") as file:
            file.write(weka_result)
        with err_file.open("w") as file:
            file.write(weka_error)


if __name__ == '__main__':
    main()
