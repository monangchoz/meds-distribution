from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from avns.utils import try_packing_custs_in_route
from problem.solution import Solution


@dataclass(order=True)
class LocalSearchArgs:
    d_cost: float
    v1: int
    v2: int

@dataclass
class SwapCustomerArgs(LocalSearchArgs):
    ci_v1: int
    ci_v2: int

class LocalSearchOperator:
    def __init__(self):
        pass

    def apply_new_route(self,
                        solution:Solution,
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
        
    def get_all_potential_args(self, solution: Solution)->Sequence[LocalSearchArgs]:
        raise NotImplementedError()

    def __call__(self, soltuion:Solution, args)->Tuple[Solution, bool]:
        raise NotImplementedError()
    
    # def do(self, original_solution: Solution, *args, **kwargs)->Solution:
    #     raise NotImplementedError()

def compute_same_route_swapping_dcost(solution:Solution, v1: int, ci_v1:int, ci_v2:int):
    original_route = solution.routes[v1]
    original_distance = solution.problem.compute_route_total_distance(original_route)
    cust_idx_1 = solution.routes[v1][ci_v1]
    cust_idx_2 = solution.routes[v1][ci_v2]
    new_route = original_route.copy()
    new_route[ci_v1] = cust_idx_2
    new_route[ci_v2] = cust_idx_1
    new_distance = solution.problem.compute_route_total_distance(new_route)
    d_cost = (new_distance-original_distance)*solution.vehicle_variable_costs[v1]
    return d_cost

def compute_swapping_dcost(solution: Solution,
                           v1: int,
                           v2: int,
                           ci_v1: int,
                           ci_v2: int)->float:
    if v1==v2:
        return compute_same_route_swapping_dcost(solution, v1, ci_v1, ci_v2)
    problem = solution.problem
    distance_matrix = solution.problem.distance_matrix
    cust_idx_1 = solution.routes[v1][ci_v1]
    cust_idx_2 = solution.routes[v2][ci_v2]
    prev_node = 0
    if len(solution.routes[v1])>1:
        prev_node = solution.routes[v1][ci_v1-1]
    next_node = 0
    if len(solution.routes[v1])>ci_v1+1:
        next_node = solution.routes[v1][ci_v1+1]
    
    d_distance_v1 = distance_matrix[prev_node, cust_idx_2]-distance_matrix[prev_node, cust_idx_1] + distance_matrix[cust_idx_2, next_node] - distance_matrix[cust_idx_1, next_node]
    prev_node = 0
    if len(solution.routes[v2])>1:
        prev_node = solution.routes[v2][ci_v2-1]
    next_node = 0
    if len(solution.routes[v2])>ci_v2+1:
        next_node = solution.routes[v2][ci_v2+1]
    d_distance_v2 = distance_matrix[prev_node, cust_idx_1]-distance_matrix[prev_node, cust_idx_2] + distance_matrix[cust_idx_1, next_node] - distance_matrix[cust_idx_2, next_node]
    d_cost = d_distance_v1*problem.vehicle_variable_costs[v1] + d_distance_v2*problem.vehicle_variable_costs[v2]
    return d_cost

def is_swapping_potential(solution: Solution,
                           v1: int,
                           v2: int,
                           ci_v1: int,
                           ci_v2: int)->bool:
    if v1==v2:
        return True
    # swapping does not violate volume and weight capacity
    cust_idx_1 = solution.routes[v1][ci_v1]
    cust_idx_2 = solution.routes[v2][ci_v2]
    problem = solution.problem
    
    
    # reefer compatibility
    if problem.node_reefer_flags[cust_idx_1] and not problem.vehicle_reefer_flags[v2]:
        return False
    if problem.node_reefer_flags[cust_idx_2] and not problem.vehicle_reefer_flags[v1]:
        return False
    
    d_filled_volumes_v1 = problem.total_demand_volumes[cust_idx_2]-problem.total_demand_volumes[cust_idx_1]
    d_filled_volumes_v2 = -d_filled_volumes_v1
    if solution.filled_volumes[v1] + d_filled_volumes_v1 > problem.vehicle_volume_capacities[v1]:
        return False
    if solution.filled_volumes[v2] + d_filled_volumes_v2 > problem.vehicle_volume_capacities[v2]:
        return False
    
    d_filled_weights_v1 = problem.total_demand_weights[cust_idx_2]-problem.total_demand_weights[cust_idx_1]
    d_filled_weights_v2 = -d_filled_weights_v1
    if solution.filled_weight_caps[v1] + d_filled_weights_v1 > problem.vehicle_volume_capacities[v1]:
        return False
    if solution.filled_weight_caps[v2] + d_filled_weights_v2 > problem.vehicle_volume_capacities[v2]:
        return False

    return True

class SwapCustomer(LocalSearchOperator):
    
    def get_all_potential_args(self, solution: Solution)->List[SwapCustomerArgs]:
        potential_args: List[SwapCustomerArgs] = []
        problem = solution.problem
        for cust_idx_1 in range(1, problem.num_nodes):
            v1 = solution.node_vhc_assignment_map[cust_idx_1]
            v1_route = solution.routes[v1]
            ci_v1 = v1_route.index(cust_idx_1)
            for cust_idx_2 in range(cust_idx_1+1, problem.num_nodes):
                v2 = solution.node_vhc_assignment_map[cust_idx_2]
                v2_route = solution.routes[v2]
                ci_v2 = v2_route.index(cust_idx_2)
                d_cost = compute_swapping_dcost(solution, v1, v2, ci_v1, ci_v2)
                if d_cost >= 0:
                    continue
                if is_swapping_potential(solution, v1, v2, ci_v1, ci_v2):
                    potential_args.append(SwapCustomerArgs(d_cost, v1, v2, ci_v1, ci_v2))
        return potential_args

    def __call__(self, solution:Solution, args: SwapCustomerArgs)->Tuple[Solution, bool]:
        return self.do(solution, args.v1, args.v2, args.ci_v1, args.ci_v2)

    def do_same_route(self, original_solution:Solution,
                 v1: int,
                 ci_v1: int,
                 ci_v2: int)->Tuple[Solution, bool]:
        solution = original_solution.copy()
        cust_idx_v1 = solution.routes[v1][ci_v1]
        cust_idx_v2 = solution.routes[v1][ci_v2]
        
        new_route = solution.routes[v1].copy()
        new_route[ci_v1] = cust_idx_v2
        new_route[ci_v2] = cust_idx_v1
        solution, is_new_route_applicable = self.apply_new_route(solution, v1, new_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True

    def do(self, original_solution:Solution,
                 v1: int,
                 v2: int,
                 ci_v1: int,
                 ci_v2: int)->Tuple[Solution, bool]:
        if v1==v2:
            return self.do_same_route(original_solution,v1,ci_v1,ci_v2)
        
        solution = original_solution.copy()
        cust_idx_v1 = solution.routes[v1][ci_v1]
        cust_idx_v2 = solution.routes[v2][ci_v2]
        
        new_v1_route = solution.routes[v1].copy()
        new_v1_route[ci_v1] = cust_idx_v2
        new_v2_route = solution.routes[v2].copy()
        new_v2_route[ci_v2] = cust_idx_v1

        solution, is_new_route_applicable = self.apply_new_route(solution, v1, new_v1_route)
        if not is_new_route_applicable:
            return original_solution, False
        solution, is_new_route_applicable = self.apply_new_route(solution, v2, new_v2_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True
    
    def __repr__(self):
        return "customer-swap"
    
@dataclass
class CustomerShiftArgs(LocalSearchArgs):
    ci_v1: int
    new_pos_in_v2: int


def compute_shifting_dcost(solution: Solution,
                           v1: int,
                           v2: int,
                           ci_v1: int,
                           new_pos_in_v2: int)->float:
    problem = solution.problem
    distance_matrix = solution.problem.distance_matrix
    cust_idx_1 = solution.routes[v1][ci_v1]
    prev_node = 0
    if len(solution.routes[v1])>1:
        prev_node = solution.routes[v1][ci_v1-1]
    next_node = 0
    if len(solution.routes[v1])>ci_v1+1:
        next_node = solution.routes[v1][ci_v1+1]
    d_distance_v1 = - distance_matrix[prev_node, cust_idx_1] - distance_matrix[cust_idx_1, next_node]
    
    prev_node = 0
    if len(solution.routes[v2])>1:
        prev_node = solution.routes[v2][new_pos_in_v2-1]
    next_node = 0
    if len(solution.routes[v2])>new_pos_in_v2:
        next_node = solution.routes[v2][new_pos_in_v2]
        
    d_distance_v2 = distance_matrix[prev_node, cust_idx_1] + distance_matrix[cust_idx_1, next_node] - distance_matrix[prev_node,next_node]
    d_cost = d_distance_v1*problem.vehicle_variable_costs[v1] + d_distance_v2*problem.vehicle_variable_costs[v2]
    if len(solution.routes[v2])==0:
        d_cost += problem.vehicle_fixed_costs[v2]
    if len(solution.routes[v1])==1 and v1 != v2:
        d_cost -= problem.vehicle_container_dims[v1]
    
    return d_cost

def is_shifting_potential(solution: Solution,
                          v1: int,
                          v2: int,
                          ci_v1: int)->bool:
    if v1==v2:
        return True

    cust_idx_1 = solution.routes[v1][ci_v1]
    problem = solution.problem

    # reefer compatibility
    if problem.node_reefer_flags[cust_idx_1] and not problem.vehicle_reefer_flags[v2]:
        return False
    
    if solution.filled_volumes[v2] + problem.total_demand_volumes[cust_idx_1] > solution.vehicle_volume_capacities[v2]:
        return False
    if solution.filled_weight_caps[v2] + problem.total_demand_weights[cust_idx_1] > solution.vehicle_weight_capacities[v2]:
        return False
    return True
    
class CustomerShift(LocalSearchOperator):
    def get_all_potential_args(self, solution: Solution)->List[CustomerShiftArgs]:
        potential_args: List[CustomerShiftArgs] = []
        problem = solution.problem
        for cust_idx_1 in range(1, problem.num_nodes):
            v1 = solution.node_vhc_assignment_map[cust_idx_1]
            ci_v1 = solution.routes[v1].index(cust_idx_1)
            for v2 in range(problem.num_vehicles):
                v2_route = solution.routes[v2]
                for new_pos_in_v2 in range(len(v2_route)+1):
                    if v2==v1 and new_pos_in_v2==ci_v1:
                        continue
                    d_cost = compute_shifting_dcost(solution, v1, v2, ci_v1, new_pos_in_v2)
                    if d_cost >= 0:
                        continue
                    if is_shifting_potential(solution, v1, v2, ci_v1):
                        potential_args.append(CustomerShiftArgs(d_cost, v1, v2, ci_v1, new_pos_in_v2))
        return potential_args

    def __call__(self, solution: Solution, args: CustomerShiftArgs):
        return self.do(solution, args.v1, args.v2, args.ci_v1, args.new_pos_in_v2)

    def do(self, original_solution:Solution, v1: int, v2: int, ci_v1: int, new_pos_in_v2: int)->Solution:
        solution = original_solution.copy()
        cust_idx = original_solution.routes[v1][ci_v1]
        
        v1_route = original_solution.routes[v1]
        new_v1_route = v1_route.copy()
        new_v1_route = new_v1_route[ci_v1]
        new_v2_route = original_solution.routes[v2].copy()
        new_v2_route = new_v2_route[:new_pos_in_v2] + [cust_idx] + new_v2_route[new_pos_in_v2:]

        solution, is_new_route_applicable = self.apply_new_route(solution, v1, new_v1_route)
        if not is_new_route_applicable:
            return original_solution
        solution, is_new_route_applicable = self.apply_new_route(solution, v2, new_v2_route)
        if not is_new_route_applicable:
            return original_solution
        return solution