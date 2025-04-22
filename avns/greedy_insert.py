from problem.solution import Solution
import numpy as np


def greedy_insert(solution:Solution, 
                  is_visited:np.ndarray)->bool:
    if np.all(is_visited):
        return True
    problem = solution.problem
    unvisited_custs_idx = np.where(np.logical_not(is_visited))[0]
    insertion_costs = []
    vehicles_idx = []
    custs_idx = []
    for vi, route in enumerate(solution.routes):
        for cust_idx in unvisited_custs_idx:
            if problem.node_reefer_flags[cust_idx] and not (problem.vehicle_reefer_flags[vi]):
                continue
            if problem.total_demand_weights[cust_idx] + solution.filled_weight_caps[vi] > solution.vehicle_weight_capacities[vi]:
                continue
            if problem.total_demand_volumes[cust_idx] + solution.filled_volumes[vi] > solution.vehicle_volume_capacities[vi]:
                continue
            insertion_cost = 0
            prev_node = 0
            if len(route)>0:
                prev_node = route[-1]
            