from typing import List, Tuple

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from problem.solution import Solution

def try_packing_custs_in_route(solution: Solution, 
                               vi: int, 
                               route:List[int])->Tuple[np.ndarray, np.ndarray, bool]:
    problem = solution.problem
    total_num_items = np.sum(solution.node_num_items[route])
            
    # this all actually can be pre-allocated in the problem interface
    # and used freely, to remove allocation time
    item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
    item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    n = 0
    for i, cust_idx in enumerate(route):
        c_num_items = solution.node_num_items[cust_idx]
        item_mask = problem.node_item_mask[cust_idx, :]
        item_dims[n:n+c_num_items] = problem.item_dims[item_mask]
        item_volumes[n:n+c_num_items] = problem.item_volumes[item_mask]
        item_weights[n:n+c_num_items] = problem.item_weights[item_mask]
        item_priorities[n:n+c_num_items] = i
        n += c_num_items
            
        # let's try packing
    container_dim = problem.vehicle_container_dims[vi]
    packing_result = random_slpack(item_dims,
                                    item_volumes,
                                    item_priorities,
                                    container_dim,
                                    0.8,
                                    5)
    return packing_result

def apply_new_route(solution:Solution,
                    vehicle_idx: int,
                    new_route: List[int])->Tuple[Solution, bool]:
    problem = solution.problem
    original_route = solution.routes[vehicle_idx].copy()
    # if infeasible, return false
    packing_result = try_packing_custs_in_route(solution, vehicle_idx, new_route)
    positions, rotations, is_packing_feasible = packing_result
    if not is_packing_feasible:
        return solution, False
    
    solution.node_vhc_assignment_map[new_route] = vehicle_idx
    solution.filled_volumes[vehicle_idx] = np.sum(problem.total_demand_volumes[new_route])
    solution.filled_weight_caps[vehicle_idx] = np.sum(problem.total_demand_weights[new_route])
    solution.routes[vehicle_idx] = new_route
    n = 0
    for i, cust_idx in enumerate(new_route):
        c_num_items = solution.node_num_items[cust_idx]
        item_mask = problem.node_item_mask[cust_idx, :]
        solution.item_positions[item_mask] = positions[n:n+c_num_items]
        solution.item_rotations[item_mask] = rotations[n:n+c_num_items]
        n += c_num_items

    d_distance = problem.compute_route_total_distance(new_route) - problem.compute_route_total_distance(original_route)
    d_cost = d_distance*problem.vehicle_variable_costs[vehicle_idx]
    if len(new_route) == 0 and len(original_route) > 0:
        solution.total_vehicle_fixed_cost -= problem.vehicle_fixed_costs[vehicle_idx]
    elif len(new_route) >0 and len(original_route)==0:
        solution.total_vehicle_fixed_cost += problem.vehicle_fixed_costs[vehicle_idx]
    solution.total_vehicle_variable_cost += d_cost
    return solution, True