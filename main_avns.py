import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from avns.greedy_insert import greedy_insert
from problem.hvrp3l import HVRP3L
from problem.solution import Solution


def run():
    filename = "JK2_nc_30_ni__914_nv_4_0.json"
    # filename = "JK2_nc_50_ni__2763_nv_4_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    initial_solution = greedy_insert(problem)
    print(initial_solution.is_feasible)

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()