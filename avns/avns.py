import time
from typing import Optional, Sequence

from avns.diversification import Diversification
from avns.saving import saving
from avns.local_search import local_search
from avns.local_search_operator import LocalSearchOperator
from avns.shake_operators import ShakeOperator
from line_profiler import profile
from problem.hvrp3l import HVRP3L
from problem.solution import Solution


class AVNS:
    def __init__(self,
                 max_iteration: int,
                 patience: int,
                 local_search_operators: Sequence[LocalSearchOperator],
                 shake_operators: Sequence[ShakeOperator],
                 max_time:int=3600):
        self.diversification: Diversification
        self.curr_solution: Solution
        self.best_solution: Solution
        self.local_search_operators: Sequence[LocalSearchOperator] = local_search_operators
        self.shake_operators: Sequence[ShakeOperator] = shake_operators
        self.max_iteration = max_iteration
        self.patience = patience
        self.non_imp_iteration: int
        self.max_time = max_time

    def reset(self, problem: HVRP3L):
        self.diversification = Diversification(problem.num_nodes)
        self.curr_solution = saving(problem)
        self.best_solution = self.curr_solution.copy()
        self.diversification.update_improvement_status(self.curr_solution)
        self.non_imp_iteration = 0

    @profile
    def solve(self, problem:HVRP3L)->Solution:
        start_time = time.time()
        self.reset(problem)
        for iteration in range(self.max_iteration):
            print(f"Iteration {iteration}, Best Total Cost: {self.best_solution.total_cost}, total distance:{self.best_solution.total_distance}")
            new_solution = self.curr_solution.copy()
            for shake_op in self.shake_operators:
                new_solution = shake_op(new_solution)
                new_solution = local_search(new_solution, self.local_search_operators)
                if new_solution.total_cost < self.best_solution.total_cost:
                    self.curr_solution = new_solution
                    self.best_solution = self.curr_solution.copy()
                    self.non_imp_iteration = 0
                else:
                    self.non_imp_iteration += 1
                if self.non_imp_iteration >=self.patience:
                    break
            self.diversification.update_improvement_status(self.best_solution)
            div_solution = self.best_solution.copy()
            div_solution = self.diversification(div_solution)
            self.curr_solution = div_solution
            current_time = time.time()-start_time
            if current_time > self.max_time:
                break
        if self.best_solution.total_cost > self.curr_solution.total_cost:
            self.best_solution = self.curr_solution
        return self.best_solution
