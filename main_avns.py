import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from avns.greedy_insert import greedy_insert
from avns.local_search import local_search
from avns.local_search_operator import (CustomerShift, RouteInterchange,
                                        SwapCustomer)
from avns.shake_operators import SE
from problem.hvrp3l import HVRP3L
from problem.solution import Solution


def run():
    filename = "JK2_nc_30_ni__914_nv_4_0.json"
    # filename = "JK2_nc_50_ni__2763_nv_4_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    initial_solution = greedy_insert(problem)
    # se21 = SE(3, True, 10, 2)
    ls_operators = [SwapCustomer(), CustomerShift(), RouteInterchange()]
    solution = local_search(initial_solution, ls_operators)

    # solution = se21(initial_solution)
    # print(solution.is_feasible)
    
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()