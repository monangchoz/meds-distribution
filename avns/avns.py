from typing import Optional, Sequence

from avns.diversification import Diversification
from avns.greedy_insert import greedy_insert
from avns.local_search import local_search
from avns.local_search_operator import LocalSearchOperator
from avns.shake_operators import ShakeOperator
from problem.hvrp3l import HVRP3L
from problem.solution import Solution


class AVNS:
    def __init__(self,
                 max_iteration: int, 
                 local_search_operators: Sequence[LocalSearchOperator],
                 shake_operators: Sequence[ShakeOperator]):
        self.diversification: Diversification
        self.curr_solution: Optional[Solution]
        self.best_solution: Optional[Solution]
        self.local_search_operators: Sequence[LocalSearchOperator] = local_search_operators
        self.shake_operators: Sequence[ShakeOperator] = shake_operators
        self.max_iteration = max_iteration

    def reset(self, problem: HVRP3L):
        self.diversification = Diversification(problem.num_nodes)
        self.curr_solution = None
        self.best_solution = None
    
    def solve(self, problem:HVRP3L)->Solution:
        self.reset(problem)
        self.curr_solution = greedy_insert(problem)
        self.best_solution = self.curr_solution.copy()
        self.diversification.update_improvement_status(self.curr_solution)

        for iteration in range(self.max_iteration):
            print(f"Iteration {iteration}, Best Total Cost: {self.best_solution.total_cost}")
            new_solution = self.curr_solution.copy()
            for shake_op in self.shake_operators:
                new_solution = shake_op(new_solution)
                new_solution = local_search(new_solution, self.local_search_operators)
                if new_solution.total_cost < self.best_solution.total_cost:
                    self.curr_solution = new_solution
                    self.best_solution = self.curr_solution.copy()
                    
            self.diversification.update_improvement_status(self.best_solution)
            div_solution = self.best_solution.copy()
            div_solution = self.diversification(div_solution)
            self.curr_solution = div_solution
        if self.best_solution.total_cost > self.curr_solution.total_cost:
            self.best_solution = self.curr_solution
        exit()
        return self.best_solution
