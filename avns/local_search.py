
import math
from multiprocessing import Pool, Manager
from multiprocessing.managers import DictProxy
from typing import List, Sequence

import numpy as np
from avns.local_search_operator import LocalSearchArgs, LocalSearchOperator
from line_profiler import profile
from problem.solution import Solution



def try_operator_parallel(ls_operator:LocalSearchOperator, args: LocalSearchArgs, shared_dict: DictProxy):
    if "chosen_args" in shared_dict:
        return
    original_solution = shared_dict["original_solution"]
    new_solution, is_ls_feasible = ls_operator(original_solution, args)
    if is_ls_feasible and "chosen_args" not in shared_dict:
        shared_dict["new_solution"] = new_solution
        shared_dict["chosen_args"] = args
            

@profile
def local_search(solution: Solution, operators: List[LocalSearchOperator])->Solution:
    available_args: List[List[LocalSearchArgs]] = [[] for _ in range(len(operators))]
    is_operator_available: np.ndarray = np.ones((len(operators),), dtype=bool)
    iteration = 0
    pool = Pool(4)
    manager = Manager()
    shared_dict = manager.dict()
    while np.any(is_operator_available):
        shared_dict.clear()
        shared_dict["original_solution"] = solution
        chosen_operator_idx = np.random.choice(np.where(is_operator_available)[0], size=1).item()
        chosen_operator = operators[chosen_operator_idx]
        trial_mode = False
        if len(available_args[chosen_operator_idx])==0:
            available_args[chosen_operator_idx] = chosen_operator.get_all_potential_args(solution)
            trial_mode = True
            if len(available_args[chosen_operator_idx])==0:
                is_operator_available[chosen_operator_idx] = False
                continue
            available_args[chosen_operator_idx] = sorted(available_args[chosen_operator_idx])
        chosen_args=None
        
        # parallelize
        batch_size = 8
        num_batches = math.ceil(len(available_args[chosen_operator_idx])/batch_size)
        for _ in range(num_batches):
            parallel_args = []
            for _ in range(batch_size):
                args = available_args[chosen_operator_idx].pop(0)
                parallel_args += [(chosen_operator, args, shared_dict)]
                if len(available_args[chosen_operator_idx]) == 0:
                    break    
            pool.starmap(try_operator_parallel, parallel_args)
            if "chosen_args" in shared_dict:
                solution = shared_dict["new_solution"]
                chosen_args = shared_dict["chosen_args"]
        # while len(available_args[chosen_operator_idx])>0:
         
        #     solution, is_ls_feasible = chosen_operator(solution, args)
        #     if is_ls_feasible:
        #         print(chosen_operator, args, solution.is_feasible)
        #         chosen_args = args
        #         break
        if chosen_args is not None:
            for i in range(len(operators)):
                for j in reversed(range(len(available_args[i]))):
                    args = available_args[i][j]
                    if args.v1 in [chosen_args.v1, chosen_args.v2] or args.v2 in [chosen_args.v1, chosen_args.v2]:
                        del available_args[i][j]
        elif trial_mode:
            is_operator_available[chosen_operator_idx] = False
        print(f"Local search iteration {iteration}, new objective: {solution.total_cost}")
        iteration += 1
        # if iteration==20:
        #     exit()
    return solution
        
        