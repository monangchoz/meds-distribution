import numpy as np

from problem.solution import Solution, NO_VEHICLE

def repair_arr1(solution: Solution):
    unvisited_customer_idxs: np.ndarray = np.nonzero(solution.cust_vhc_assignment_map!=NO_VEHICLE)[0]
    for ci in unvisited_customer_idxs:
        # try to visit to a vehicle, if not possible, then 
        # return infeasible?
        