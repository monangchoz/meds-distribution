import pathlib
import random
import time


import numpy as np
from problem.hvrp3l import HVRP3L
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo_interface.arr1 import ARR1
from pymoo_interface.hvrp3l_opt import HVRP3L_OPT


def run():
    # filename = "JK2_nc_30_ni__914_nv_4_0.json"
    filename = "JK2_nc_50_ni__2763_nv_4_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    start = time.time()
    problem_intf = HVRP3L_OPT(problem, ARR1(problem.num_customers, problem.num_vehicles))
    algo = DE(pop_size=10)
    termination = DefaultSingleObjectiveTermination(n_max_gen=100)
    res = minimize(problem_intf, algo, termination=termination, 
                   seed=1, 
                   verbose=True)
    end = time.time()
    print(end-start)
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()