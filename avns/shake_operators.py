import random
from itertools import combinations
from typing import List, Tuple

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from problem.solution import Solution


def swap_customers_in_routes(v1_route: List[int], 
                             v1_custs_idx: List[int], 
                             v2_route:List[int], 
                             v2_custs_idx: List[int]) -> Tuple[List[int], List[int]]: 
    new_v1_route, new_v2_route = v1_route.copy(), v2_route.copy()
    cust_pos_in_v1: List[int] = []
    cust_pos_in_v2: List[int] = []
    for ci, cust_idx in enumerate(v1_route):
        if cust_idx in v1_custs_idx:
            cust_pos_in_v1.append(ci)
            
    for ci, cust_idx in enumerate(v2_route):
        if cust_idx in v2_custs_idx:
            cust_pos_in_v2.append(ci)

    for i, pos in enumerate(cust_pos_in_v1):
        new_v1_route[pos] = v2_custs_idx[i]
    for i, pos in enumerate(cust_pos_in_v2):
        new_v2_route[pos] = v1_custs_idx[i]
    return new_v1_route, new_v2_route

def try_packing_custs_in_route(solution: Solution, vi: int, route:List[int]):
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


class ShakeOperator:
    def __init__(self):
        pass

class SE(ShakeOperator):
    def __init__(self, 
                 max_segment_length:int, 
                 fixed_segment_length:bool,
                 max_trials:int,
                 num_applications:int):
        super().__init__()
        self.max_segment_length = max_segment_length
        self.fixed_segment_length = fixed_segment_length
        self.max_trials = max_trials
        self.num_applications = num_applications

    def do_once(self, original_solution: Solution)->Solution:
        non_empty_routes_idx = [vi for vi in range(original_solution.num_vehicles) if len(original_solution.routes[vi]) > 0]
        if len(non_empty_routes_idx) == 1:
            return original_solution
        
        solution: Solution = original_solution.copy()
        problem = solution.problem
        segment_length = self.max_segment_length
        for trial in range(self.max_trials):
            if not self.fixed_segment_length:
                segment_length = random.randint(1, self.max_segment_length)
            # select two random routes
            random.shuffle(non_empty_routes_idx)
            v1, v2 = non_empty_routes_idx[:2]
            if len(solution.routes[v1]) < segment_length or len(solution.routes[v2]) < segment_length:
                continue
            
            # select random segment from v1
            start_idx = random.randint(0, len(original_solution.routes[v1])-segment_length)
            v1_custs_idx = original_solution.routes[v1][start_idx:start_idx+segment_length]
            candidate_v2_custs_idx = combinations(original_solution.routes[v2], segment_length)
            for v2_custs_idx_tuple in candidate_v2_custs_idx:
                v2_custs_idx = list(v2_custs_idx_tuple)
                # try swapping
                # if feasible then return
                # else continue
                d_filled_volumes_v1 = np.sum(problem.total_demand_volumes[v2_custs_idx] - problem.total_demand_volumes[v1_custs_idx])
                d_filled_volumes_v2 = -d_filled_volumes_v1
                if solution.filled_volumes[v1] + d_filled_volumes_v1 > problem.vehicle_volume_capacities[v1]:
                    continue
                if solution.filled_volumes[v2] + d_filled_volumes_v2 > problem.vehicle_volume_capacities[v2]:
                    continue
 
                d_filled_weights_v1 = np.sum(problem.total_demand_weights[v2_custs_idx] - problem.total_demand_weights[v1_custs_idx])
                d_filled_weights_v2 = -d_filled_weights_v1
                if solution.filled_weight_caps[v1] + d_filled_weights_v1 > problem.vehicle_weight_capacities[v1]:
                    continue
                if solution.filled_weight_caps[v2] + d_filled_weights_v2 > problem.vehicle_weight_capacities[v2]:
                    continue

                new_v1_route, new_v2_route = swap_customers_in_routes(original_solution.routes[v1], 
                                                                      v1_custs_idx,
                                                                      original_solution.routes[v2],
                                                                      v2_custs_idx)
                print(v1_custs_idx, solution.routes[v1], new_v1_route)
                print(v2_custs_idx, solution.routes[v2], new_v2_route)
                
                packing_result_v1 = try_packing_custs_in_route(solution, v1, new_v1_route)
                positions_v1, rotations_v1, is_packing_feasible_v1 = packing_result_v1
                if not is_packing_feasible_v1:
                    continue
                packing_result_v2 = try_packing_custs_in_route(solution, v2, new_v2_route)
                positions_v2, rotations_v2, is_packing_feasible_v2 = packing_result_v2
                if not is_packing_feasible_v2:
                    continue
                
                # swapping is feasible, commit the route
                solution.node_vhc_assignment_map[new_v1_route] = v1
                solution.filled_volumes[v1] += d_filled_volumes_v1
                solution.filled_weight_caps[v1] += d_filled_weights_v1
                solution.routes[v1] = new_v1_route
                n = 0
                for i, cust_idx in enumerate(new_v1_route):
                    c_num_items = solution.node_num_items[cust_idx]
                    item_mask = problem.node_item_mask[cust_idx, :]
                    solution.item_positions[item_mask] = positions_v1[n:n+c_num_items]
                    solution.item_rotations[item_mask] = rotations_v1[n:n+c_num_items]
                    n += c_num_items
                    
                solution.node_vhc_assignment_map[new_v2_route] = v2
                solution.filled_volumes[v2] += d_filled_volumes_v2
                solution.filled_weight_caps[v2] += d_filled_weights_v2
                solution.routes[v2] = new_v2_route
                n = 0
                for i, cust_idx in enumerate(new_v2_route):
                    c_num_items = solution.node_num_items[cust_idx]
                    item_mask = problem.node_item_mask[cust_idx, :]
                    solution.item_positions[item_mask] = positions_v2[n:n+c_num_items]
                    solution.item_rotations[item_mask] = rotations_v2[n:n+c_num_items]
                    n += c_num_items

                # compute d_variable costs
                d_distance_v1 = problem.compute_route_total_distance(new_v1_route) - problem.compute_route_total_distance(original_solution.routes[v1])
                d_distance_v2 = problem.compute_route_total_distance(new_v2_route) - problem.compute_route_total_distance(original_solution.routes[v2])
                d_cost = d_distance_v1*problem.vehicle_variable_costs[v1] + d_distance_v2*problem.vehicle_variable_costs[v2]
                solution.total_vehicle_variable_cost += d_cost
                return solution
        return original_solution

    def __call__(self, original_solution: Solution)->Solution:
        solution = original_solution
        for n in range(self.num_applications):
            solution = self.do_once(solution)
        return solution