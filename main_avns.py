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


def setup_avns(max_iteration:int)->AVNS:
    ls_operators = [CustomerShift(), RouteInterchange(), SwapCustomer()]
    sse21 = SE(2, True, 5, 1)
    sse31 = SE(3, True, 5, 1)
    sse22 = SE(2, True, 5, 2)
    sse32 = SE(3, True, 5, 2)
    vse1 = SE(3, False, 5, 1)
    vse2 = SE(3, False, 5, 2)
    shake_operators = [sse21,sse31,sse22,sse32,vse1,vse2]
    avns = AVNS(max_iteration,ls_operators,shake_operators)
    return avns


def run():
    # filename = "JK2_nc_30_ni__914_nv_4_0.json"
    # filename = "JK2_nc_50_ni__2763_nv_4_0.json"
    filename = "JK2_nc_50_ni__3719_nv_10_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    avns = setup_avns(100)
    solution = avns.solve(problem)
    solution.is_feasible
    
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()