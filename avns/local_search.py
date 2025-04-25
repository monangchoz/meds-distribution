from typing import List, Sequence

import numpy as np
from avns.local_search_operator import LocalSearchArgs, LocalSearchOperator
from problem.solution import Solution


def local_search(solution: Solution, operators: List[LocalSearchOperator])->Solution:
    available_args: List[List[LocalSearchArgs]] = [[] for _ in range(len(operators))]
    is_operator_available: np.ndarray = np.ones((len(operators),), dtype=bool)
    iteration = 0
    while np.any(is_operator_available):
        chosen_operator_idx = np.random.choice(np.where(is_operator_available)[0], size=1).item()
        chosen_operator = operators[chosen_operator_idx]
        if len(available_args[chosen_operator_idx])==0:
            available_args[chosen_operator_idx] = chosen_operator.get_all_potential_args(solution)
            if len(available_args[chosen_operator_idx])==0:
                is_operator_available[chosen_operator_idx] = False
                continue
            available_args[chosen_operator_idx] = sorted(available_args[chosen_operator_idx])
        chosen_args=None
        while len(available_args[chosen_operator_idx])>0:
            args = available_args[chosen_operator_idx].pop(0)
            solution, is_ls_feasible = chosen_operator(solution, args)
            # solution.is_feasible 
            if is_ls_feasible:
                chosen_args = args
                break
        if chosen_args is not None:
            for i in range(len(operators)):
                for j in reversed(range(len(available_args[i]))):
                    args = available_args[i][j]
                    if args.v1 in [chosen_args.v1, chosen_args.v2] or args.v2 in [chosen_args.v1, chosen_args.v2]:
                        del available_args[i][j]
        print(f"Local search iteration {iteration}, new objective: {solution.total_cost}")
        iteration += 1
    return solution
        
        