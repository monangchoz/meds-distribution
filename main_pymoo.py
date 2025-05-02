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
from pymoo_interface.arr2 import ARR2
from pymoo_interface.hvrp3l_opt import (HVRP3L_OPT, DuplicateElimination,
                                        RepairEncoding)

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


def setup_algorithm(algo_name: str, problem:HVRP3L):
    if algo_name == "brkga":
        algo = BRKGA(n_elites=20, 
                    n_offsprings=10, 
                    n_mutants=5,
                    repair=RepairEncoding(), 
                    eliminate_duplicates=DuplicateElimination(problem))
    return algo

def run():
    args = parse_args()
    filename = args.filename
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    start_time = time.time()
    pool = mp.Pool(8)
    runner = StarmapParallelization(pool.starmap)
    algo = setup_algorithm(args.algo_name, problem)
    problem_intf = HVRP3L_OPT(problem, ARR2(problem.num_customers, problem.num_vehicles), elementwise_runner=runner)
    
    termination = DefaultSingleObjectiveTermination(n_max_gen=10)
    res = minimize(problem_intf, algo, termination=termination,
                   seed=1,
                   verbose=True)
    solution = problem_intf.decode(res.X)
    end_time = time.time()
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()