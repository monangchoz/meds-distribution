import numpy as np

from problem.solution import Solution, NO_VEHICLE

class RepairMechanism:
    def repair(self, solution: Solution):
        raise NotImplementedError

class ARR1(RepairMechanism):
    def repair(self, solution: Solution):
        # print(solution.cust_vhc_assignment_map)
        unvisited_customer_idxs: np.ndarray = np.nonzero(solution.cust_vhc_assignment_map==NO_VEHICLE)[0]
        if len(unvisited_customer_idxs) == 0:
            return
        print("HELLO", unvisited_customer_idxs)
        exit()
    # for ci in unvisited_customer_idxs:
    #     # try to visit to a vehicle, if not possible, then 
    #     # return infeasible?
        