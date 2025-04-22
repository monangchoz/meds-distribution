import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from problem.hvrp3l import HVRP3L
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo_interface.arr1 import ARR1
from pymoo_interface.hvrp3l_opt import HVRP3L_OPT, DuplicateElimination


def run():
    filename = "JK2_nc_30_ni__914_nv_4_0.json"
    # filename = "JK2_nc_50_ni__2763_nv_4_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    start = time.time()
    pool = mp.Pool(6)
    runner = StarmapParallelization(pool.starmap)
    problem_intf = HVRP3L_OPT(problem, ARR1(problem.num_customers, problem.num_vehicles), elementwise_runner=runner)
    algo = BRKGA(n_elites=20, n_offsprings=10, n_mutants=5, eliminate_duplicates=DuplicateElimination(problem))
    termination = DefaultSingleObjectiveTermination(n_max_gen=10)
    res = minimize(problem_intf, algo, termination=termination,
                   seed=1,
                   verbose=True)
    solution = problem_intf.decode(res.X)
    
    end = time.time()
    print(end-start)
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()