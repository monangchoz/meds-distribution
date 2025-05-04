from typing import Tuple

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from problem.hvrp3l import HVRP3L
from problem.solution import NO_VEHICLE, Solution


def try_inserting_to_vehicles(solution:Solution, sorted_custs_idx: np.ndarray)->Tuple[Solution, bool]:
    """

    """
    problem = solution.problem
    # ncu = 0
    if np.all(solution.node_vhc_assignment_map != NO_VEHICLE):
        print("HELLO")
        # print(ncu)
        # try to pack here actually, not after every insertion..?
        # check if packing feasible
        for vehicle_idx, route in enumerate(solution.routes):
            if len(route)==0:
                continue
            total_num_items = np.sum(problem.node_num_items[route])
            item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
            item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=int)
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
            container_dim = problem.vehicle_container_dims[vehicle_idx]
            packing_result = random_slpack(item_dims,
                                            item_volumes,
                                            item_priorities,
                                            container_dim,
                                            0.8,
                                            5)
            positions, rotations, is_packing_feasible = packing_result
            if not is_packing_feasible:
                return solution, False
            n = 0
            for i, cust_idx in enumerate(route):
                c_num_items = solution.node_num_items[cust_idx]
                item_mask = problem.node_item_mask[cust_idx, :]
                solution.item_positions[item_mask] = positions[n:n+c_num_items]
                solution.item_rotations[item_mask] = rotations[n:n+c_num_items]
                n += c_num_items
        return solution, True
    
    new_cust_idx = None
    for cust_idx in sorted_custs_idx:
        if solution.node_vhc_assignment_map[cust_idx] == NO_VEHICLE:
            new_cust_idx = cust_idx
            break

    insertion_costs = []
    vehicles_idx = []
    for vi, route in enumerate(solution.routes):
        if problem.node_reefer_flags[new_cust_idx] and not (problem.vehicle_reefer_flags[vi]):
            continue
        if problem.total_demand_weights[new_cust_idx] + solution.filled_weight_caps[vi] > solution.vehicle_weight_capacities[vi]:
            continue
        if problem.total_demand_volumes[new_cust_idx] + solution.filled_volumes[vi] > solution.vehicle_volume_capacities[vi]:
            continue
        insertion_cost = 0
        prev_node = 0
        if len(route)>0:
            prev_node = route[-1]
        distance = problem.distance_matrix[prev_node, new_cust_idx] + problem.distance_matrix[new_cust_idx, 0] - problem.distance_matrix[prev_node, 0]
        insertion_cost += distance*problem.vehicle_variable_costs[vi]
        if len(route)==0:
            insertion_cost += problem.vehicle_fixed_costs[vi]
        insertion_costs.append(insertion_cost)
        vehicles_idx.append(vi)
    
    insertion_costs = np.asanyarray(insertion_costs)
    vehicles_idx = np.asanyarray(vehicles_idx)
    sorted_idx = np.argsort(insertion_costs)
    insertion_costs = insertion_costs[sorted_idx]
    vehicles_idx = vehicles_idx[sorted_idx]
    print(cust_idx, vehicles_idx)
    for vehicle_idx, cost in zip(vehicles_idx, insertion_costs):
        solution.filled_volumes += problem.total_demand_volumes[new_cust_idx]
        solution.filled_weight_caps += problem.total_demand_weights[new_cust_idx]
        solution.node_vhc_assignment_map[new_cust_idx] = vehicle_idx
        if len(solution.routes[vehicle_idx])==0:
            solution.total_vehicle_fixed_cost += problem.vehicle_fixed_costs[vehicle_idx]
            cost -= problem.vehicle_fixed_costs[vehicle_idx]
        solution.total_vehicle_variable_cost += cost
        solution.routes[vehicle_idx].append(new_cust_idx)
        
        new_solution, is_feasible_solution_found = try_inserting_to_vehicles(solution, sorted_custs_idx)
        if is_feasible_solution_found:
            return new_solution, True
        
        solution.filled_volumes -= problem.total_demand_volumes[new_cust_idx]
        solution.filled_weight_caps -= problem.total_demand_weights[new_cust_idx]
        solution.routes[vehicle_idx] = solution.routes[vehicle_idx][:-1]
        solution.node_vhc_assignment_map[new_cust_idx] = NO_VEHICLE
        if len(solution.routes[vehicle_idx])==0:
            solution.total_vehicle_fixed_cost -= problem.vehicle_fixed_costs[vehicle_idx]
        solution.total_vehicle_variable_cost -= cost
    return solution, False

def greedy_insert(problem: HVRP3L, max_trials=1)->Solution:
    initial_solution: Solution
    for trial in range(max_trials):
        # print(f"trial {trial}")
        solution = Solution(problem)
        sorted_custs_idx = np.arange(problem.num_customers)+1
        np.random.shuffle(sorted_custs_idx)
        sorted_custs_idx = sorted_custs_idx.tolist()
        initial_solution, is_feasible_solution_found = try_inserting_to_vehicles(solution, sorted_custs_idx)
        if is_feasible_solution_found:
            break
    raise ValueError()
    return initial_solution
            