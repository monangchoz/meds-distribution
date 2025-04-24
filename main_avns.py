import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from avns.greedy_insert import greedy_insert
from avns.local_search_operator import CustomerShift, SwapCustomer
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
    sc = SwapCustomer()
    potential_args = sc.get_all_potential_args(initial_solution)
    potential_args = sorted(potential_args)
    chosen_args = None
    while len(potential_args)>0:
        print("good")
        for args in potential_args:
            chosen_args = args
            solution, is_ls_feasible = sc(initial_solution, args)
            print(is_ls_feasible, solution.is_feasible)
            break
        for i in reversed(range(len(potential_args))):
            args = potential_args[i]
            if args.v1 in [chosen_args.v1, chosen_args.v2] or args.v2 in [chosen_args.v1, chosen_args.v2]:
                del potential_args[i]

    # solution = se21(initial_solution)
    # print(solution.is_feasible)
    
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()