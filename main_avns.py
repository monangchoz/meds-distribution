import argparse
import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from avns.avns import AVNS
from avns.greedy_insert import greedy_insert
from avns.local_search import local_search
from avns.local_search_operator import (CustomerShift, RouteInterchange,
                                        SwapCustomer)
from avns.shake_operators import SE
from problem.hvrp3l import HVRP3L
from problem.solution import Solution
from utils import has_valid_result


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="experiment arguments.")
    parser.add_argument("--instance-file-name",
                        type=str,
                        required=True,
                        help="instance filename")
    
    parser.add_argument("--max-iteration",
                    type=int,
                    required=True,
                    help="maximum iteration")
    
    parser.add_argument("--patience",
                    type=int,
                    required=True,
                    help="num of iteration not improving before early stopping")
    
    return parser.parse_args()


def setup_avns(max_iteration:int, patience:int)->AVNS:
    ls_operators = [CustomerShift(), RouteInterchange(), SwapCustomer()]
    sse21 = SE(2, True, 5, 1)
    sse31 = SE(3, True, 5, 1)
    sse22 = SE(2, True, 5, 2)
    sse32 = SE(3, True, 5, 2)
    vse1 = SE(3, False, 5, 1)
    vse2 = SE(3, False, 5, 2)
    shake_operators = [sse21,sse31,sse22,sse32,vse1,vse2]
    avns = AVNS(max_iteration,patience,ls_operators,shake_operators)
    return avns

def run():
    args = parse_args()
    filename = args.instance_file_name
    filename_without_extension = filename[:-5]
    instance_filepath = pathlib.Path("instances") / filename
    result_dir = pathlib.Path("results") / "avns"
    result_filepath = result_dir / f"{filename_without_extension}.csv"

    if has_valid_result(result_filepath):
        print(f"Skipping {filename}: valid result already exists.")
        return

    start_time = time.time()
    problem = HVRP3L.read_from_json(instance_filepath)
    avns = setup_avns(args.max_iteration, args.patience)
    solution = avns.solve(problem)
    end_time = time.time()
    running_time = end_time - start_time

    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_filepath.absolute(), "a") as f:
        result_str = f"{solution.total_cost},{running_time}\n"
        f.write(result_str)


        

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()