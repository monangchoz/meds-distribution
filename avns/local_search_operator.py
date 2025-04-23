import random
import numpy as np

from problem.hvrp3l import HVRP3L
from problem.solution import Solution
from avns.utils import try_packing_custs_in_route

class LocalSearchOperator:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwds):
        raise NotImplementedError()
    
class SwapCustomer(LocalSearchOperator):
    def __call__(self, original_solution:Solution)->Solution:
        problem = original_solution.problem
        solution = original_solution.copy()
        non_empty_routes_idx = [vi for vi in range(problem.num_vehicles) if len(solution.routes[vi])>0]
        v1, v2 = random.choices(non_empty_routes_idx, k=2)
        ci_v1 = random.randint(0, len(solution.routes[v1])-1)
        ci_v2 = random.randint(0, len(solution.routes[v2])-1)
        cust_idx_v1 = solution.routes[v1][ci_v1]
        cust_idx_v2 = solution.routes[v2][ci_v2]
        
        
        
        d_filled_volumes_v1 = np.sum(problem.total_demand_volumes[cust_idx_v2] - problem.total_demand_volumes[cust_idx_v1])
        d_filled_volumes_v2 = -d_filled_volumes_v1
        if solution.filled_volumes[v1] + d_filled_volumes_v1 > problem.vehicle_volume_capacities[v1]:
            return original_solution
        if solution.filled_volumes[v2] + d_filled_volumes_v2 > problem.vehicle_volume_capacities[v2]:
            return original_solution

        d_filled_weights_v1 = np.sum(problem.total_demand_weights[cust_idx_v2] - problem.total_demand_weights[cust_idx_v1])
        d_filled_weights_v2 = -d_filled_weights_v1
        if solution.filled_weight_caps[v1] + d_filled_weights_v1 > problem.vehicle_weight_capacities[v1]:
            return original_solution
        if solution.filled_weight_caps[v2] + d_filled_weights_v2 > problem.vehicle_weight_capacities[v2]:
            return original_solution
        
        new_v1_route = solution.routes[v1].copy()
        new_v1_route[ci_v1] = cust_idx_v2
        new_v2_route = solution.routes[v2].copy()
        new_v2_route[ci_v2] = cust_idx_v1
        
        packing_result_v1 = try_packing_custs_in_route(solution, v1, new_v1_route)
        positions_v1, rotations_v1, is_packing_feasible_v1 = packing_result_v1
        if not is_packing_feasible_v1:
            return original_solution
        packing_result_v2 = try_packing_custs_in_route(solution, v2, new_v2_route)
        positions_v2, rotations_v2, is_packing_feasible_v2 = packing_result_v2
        if not is_packing_feasible_v2:
            return original_solution
        
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