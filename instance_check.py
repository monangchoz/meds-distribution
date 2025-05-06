import argparse
import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from avns.saving import saving
from problem.hvrp3l import HVRP3L


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="experiment arguments.")
    parser.add_argument("--instance-file-name",
                        type=str,
                        required=True,
                        help="instance filename")
    
    return parser.parse_args()

def run():
    args = parse_args()
    filename = args.instance_file_name
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    solution = saving(problem)
    solution.is_feasible

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()