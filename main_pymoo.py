import argparse
import multiprocessing as mp
import pathlib
import random
import time

import numpy as np
from problem.hvrp3l import HVRP3L
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo_interface.arr2 import ARR2
from pymoo_interface.hvrp3l_opt import (HVRP3L_OPT, DuplicateElimination,
                                        RepairEncoding)
from pymoo.core.termination import TerminateIfAny
from pymoo.termination.max_time import TimeBasedTermination
from utils import has_valid_result


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="experiment arguments.")
    
    parser.add_argument("--algo-name",
                        type=str,
                        choices=["ga","brkga","pso","de"],
                        required=True,
                        help="algo name")
    
    parser.add_argument("--instance-file-name",
                        type=str,
                        required=True,
                        help="instance filename")
    
    
    parser.add_argument("--patience",
                    type=int,
                    required=True,
                    help="num of iteration not improving before early stopping")
    
    return parser.parse_args()


def setup_algorithm(algo_name: str, problem:HVRP3L):
    if algo_name == "brkga":
        algo = BRKGA(n_elites=30, 
                    n_offsprings=15, 
                    n_mutants=5,
                    repair=RepairEncoding(), 
                    eliminate_duplicates=DuplicateElimination(problem))
    elif algo_name == "ga":
        algo = GA(pop_size=50, 
                  repair=RepairEncoding(), 
                  eliminate_duplicates=DuplicateElimination(problem))
    elif algo_name == "de":
        algo = DE(pop_size=50, 
                  repair=RepairEncoding())
    elif algo_name == "pso":
        algo = PSO(pop_size=50, repair=RepairEncoding())
    return algo

def run():
    args = parse_args()
    filename = args.instance_file_name
    filename_without_extension = filename[:-5]
    result_dir = pathlib.Path()/"results"/args.algo_name
    result_filepath = result_dir/f"{filename_without_extension}.csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(args.algo_name, filename)
    if has_valid_result(result_filepath):
        print(f"Skipping {filename}: valid result already exists.")
        return
    
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    start_time = time.time()
    algo = setup_algorithm(args.algo_name, problem)
    
    problem_intf = HVRP3L_OPT(problem, ARR2(problem.num_customers, problem.num_vehicles))#, elementwise_runner=runner)
    termination_max_gen_period = DefaultSingleObjectiveTermination(n_max_gen=100, period=args.patience)
    termination_time = TimeBasedTermination("01:00:00")
    termination = TerminateIfAny(termination_max_gen_period, termination_time)
    res = minimize(problem_intf, algo, termination=termination,
                   seed=1,
                   verbose=True)
    solution = problem_intf.decode(res.X)
    end_time = time.time()
    running_time = end_time-start_time
    
    with open(result_filepath.absolute(), "+a") as f:
        result_str = f"{solution.total_cost},{running_time}\n"
        f.write(result_str)

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()