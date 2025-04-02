import math

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ep_heuristic.random_slpack import random_slpack
from problem.hvrp3l import HVRP3L
from problem.solution import Solution


def insert_items(solution: Solution, route: np.ndarray, vehicle_idx: int)->bool:
    
    
    item_dims = solution.item_dims
    

class HVRP3L_OPT(ElementwiseProblem):
    def __init__(self,
                 hvrp3l_instance: HVRP3L,
                 **kwargs):
        super().__init__(elementwise=True, **kwargs)
        self.hvrp3l_instance = hvrp3l_instance
        self.n_var = 2*hvrp3l_instance.num_customers 
        # first num_customers dims are for customer priority
        # the next num_customers dims are for vehicle assignment
        self.xl = np.zeros([self.n_var, ], dtype=float)
        self.xu = np.ones([self.n_var, ], dtype=float)
        
    # this is the decoding method
    def _evaluate(self, x, out, *args, **kwargs):
        solution: Solution = Solution(self.hvrp3l_instance)
        # try to map first into vehicle
        # if not feasible?
        
        for i in range(self.hvrp3l_instance.num_customers, 2*self.hvrp3l_instance.num_customers):
            ci = i-self.hvrp3l_instance.num_customers
            vi = math.floor(x[i]*self.hvrp3l_instance.num_vehicles)
            solution.cust_vhc_assignment_map[ci] = vi
        for vi in range(self.hvrp3l_instance.num_vehicles):
            vi_cust_idx = np.nonzero(solution.cust_vhc_assignment_map==vi)[0]
            if len(vi_cust_idx) == 0:
                continue
            cust_priorities = x[vi_cust_idx]
            sorted_idx = np.argsort(cust_priorities)
            vi_cust_idx = vi_cust_idx[sorted_idx]
        
        # try inserting, if not feasible, cancel this vehicle assignment, and 
        # then we repair later.
            