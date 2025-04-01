
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from problem.hvrp3l import HVRP3L
from problem.solution import Solution


class HVRP3L_OPT(ElementwiseProblem):
    def __init__(self, 
                 hvrp3l_instance: HVRP3L,
                 elementwise=True, 
                 **kwargs):
        super().__init__(elementwise, **kwargs)
        self.hvrp3l_instance = hvrp3l_instance
        self.num_dims = 2*hvrp3l_instance.num_customers 
        # first num_customers dims are for customer priority
        # the next num_customers dims are for vehicle assignment
        self.xl = np.zeros([self.num_dims, ], dtype=float)
        self.xu = np.ones([self.num_dims, ], dtype=float)
        
    # this is the decoding method
    def _evaluate(self, x, out, *args, **kwargs):
        solution: Solution = Solution(self.hvrp3l_instance)
        # try to map first into vehicle
        # if not feasible?
        for i in range(self.hvrp3l_instance.num_customers, 2*self.hvrp3l_instance.num_customers):
            print(i)
        exit()
            