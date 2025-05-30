import math
from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
from avns.utils import apply_new_route
from line_profiler import profile
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
        
    def get_all_potential_args(self, solution: Solution)->Sequence[LocalSearchArgs]:
        raise NotImplementedError()

    def __call__(self, solution:Solution, args)->Tuple[Solution, bool]:
        raise NotImplementedError()
    
    # def do(self, original_solution: Solution, *args, **kwargs)->Solution:
    #     raise NotImplementedError()

@profile
def compute_same_route_swapping_dcost(solution:Solution, v1: int, ci_v1:int, ci_v2:int):
    original_route = solution.routes[v1]
    a = min(ci_v1, ci_v2)
    b = max(ci_v1, ci_v2)
    cust_a = original_route[a]
    cust_b = original_route[b]
    route_len = len(original_route)
    prev_node_a = 0
    if a > 0:
        prev_node_a = original_route[a-1]
    next_node_a = 0
    if a+1 < route_len:
        next_node_a = original_route[a+1]
    
    prev_node_b = 0
    if b > 0:
        prev_node_b = original_route[b-1]
    next_node_b = 0
    if b+1 < route_len:
        next_node_b = original_route[b+1]
    
    new_prev_a = prev_node_b
    new_next_a = next_node_b
    new_prev_b = prev_node_a
    new_next_b = next_node_a
    if a == b-1:
        new_prev_a = cust_b
        new_next_b = cust_a
    distance_matrix = solution.problem.distance_matrix
    d_distance = distance_matrix[new_prev_a, cust_a] + distance_matrix[cust_a, new_next_a] + distance_matrix[new_prev_b, cust_b] + distance_matrix[cust_b, new_next_b]
    d_distance += -(distance_matrix[prev_node_a, cust_a] + distance_matrix[cust_a, next_node_a] + distance_matrix[prev_node_b, cust_b] + distance_matrix[cust_b, next_node_b])
    d_cost = d_distance*solution.vehicle_variable_costs[v1]

    return d_cost

@profile
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
    if ci_v1>0:
        prev_node = solution.routes[v1][ci_v1-1]
    next_node = 0
    if len(solution.routes[v1])>ci_v1+1:
        next_node = solution.routes[v1][ci_v1+1]
    
    d_distance_v1 = distance_matrix[prev_node, cust_idx_2]-distance_matrix[prev_node, cust_idx_1] + distance_matrix[cust_idx_2, next_node] - distance_matrix[cust_idx_1, next_node]
    prev_node = 0
    if ci_v2>0:
        prev_node = solution.routes[v2][ci_v2-1]
    next_node = 0
    if len(solution.routes[v2])>ci_v2+1:
        next_node = solution.routes[v2][ci_v2+1]
    d_distance_v2 = distance_matrix[prev_node, cust_idx_1]-distance_matrix[prev_node, cust_idx_2] + distance_matrix[cust_idx_1, next_node] - distance_matrix[cust_idx_2, next_node]
    d_cost = d_distance_v1*problem.vehicle_variable_costs[v1] + d_distance_v2*problem.vehicle_variable_costs[v2]
    return d_cost

@profile
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
            v1 = solution.node_vhc_assignment_map[cust_idx_1].item()
            v1_route = solution.routes[v1]
            ci_v1 = v1_route.index(cust_idx_1)
            for cust_idx_2 in range(cust_idx_1+1, problem.num_nodes):
                v2 = solution.node_vhc_assignment_map[cust_idx_2].item()
                v2_route = solution.routes[v2]
                ci_v2 = v2_route.index(cust_idx_2)
                d_cost = compute_swapping_dcost(solution, v1, v2, ci_v1, ci_v2)
                if d_cost >= -1e-9:
                    continue
                if is_swapping_potential(solution, v1, v2, ci_v1, ci_v2):
                    potential_args.append(SwapCustomerArgs(d_cost, v1, v2, ci_v1, ci_v2))
        return potential_args

    def __call__(self, solution:Solution, args: SwapCustomerArgs)->Tuple[Solution, bool]:
        return self.do(solution, args.v1, args.v2, args.ci_v1, args.ci_v2)

    @profile
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
        solution, is_new_route_applicable = apply_new_route(solution, v1, new_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True

    @profile
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

        solution, is_new_route_applicable = apply_new_route(solution, v1, new_v1_route)
        if not is_new_route_applicable:
            return original_solution, False
        solution, is_new_route_applicable = apply_new_route(solution, v2, new_v2_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True
    
    def __repr__(self):
        return "customer-swap"
    
@dataclass
class CustomerShiftArgs(LocalSearchArgs):
    ci_v1: int
    new_pos_in_v2: int

@profile
def compute_same_route_shifting_dcost(solution: Solution,
                                        v1: int,
                                        ci: int,
                                        new_pos: int)->float:
    if ci==new_pos:
        return 0
    original_route = solution.routes[v1]
    cust_idx = original_route[ci]
    new_route = original_route.copy()
    new_route = new_route[:ci] + new_route[ci+1:]
    new_route = new_route[:new_pos] + [cust_idx] + new_route[new_pos:]
    original_distance = solution.problem.compute_route_total_distance(original_route)
    new_distance = solution.problem.compute_route_total_distance(new_route)
    d_cost = (new_distance-original_distance)*solution.vehicle_variable_costs[v1]
    return d_cost

    
@profile
def compute_shifting_dcost(solution: Solution,
                           v1: int,
                           v2: int,
                           ci_v1: int,
                           new_pos_in_v2: int)->float:
    if v1==v2:
        return compute_same_route_shifting_dcost(solution, v1, ci_v1, new_pos_in_v2)
    problem = solution.problem
    distance_matrix = solution.problem.distance_matrix
    cust_idx_1 = solution.routes[v1][ci_v1]
    prev_node = 0
    if ci_v1>0:
        prev_node = solution.routes[v1][ci_v1-1]
    next_node = 0
    if len(solution.routes[v1])>ci_v1+1:
        next_node = solution.routes[v1][ci_v1+1]
    d_distance_v1 = distance_matrix[prev_node, next_node] - distance_matrix[prev_node, cust_idx_1] - distance_matrix[cust_idx_1, next_node]
    
    prev_node = 0
    if new_pos_in_v2>0:
        prev_node = solution.routes[v2][new_pos_in_v2-1]
    next_node = 0
    if len(solution.routes[v2])>new_pos_in_v2:
        next_node = solution.routes[v2][new_pos_in_v2]
        
    d_distance_v2 = distance_matrix[prev_node, cust_idx_1] + distance_matrix[cust_idx_1, next_node] - distance_matrix[prev_node,next_node]
    d_cost = d_distance_v1*problem.vehicle_variable_costs[v1] + d_distance_v2*problem.vehicle_variable_costs[v2]
    if len(solution.routes[v2])==0:
        d_cost += problem.vehicle_fixed_costs[v2]
    if len(solution.routes[v1])==1 and v1 != v2:
        d_cost -= problem.vehicle_fixed_costs[v1]
    return d_cost

@profile
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
            v1 = solution.node_vhc_assignment_map[cust_idx_1].item()
            ci_v1 = solution.routes[v1].index(cust_idx_1)
            for v2 in range(problem.num_vehicles):
                v2_route = solution.routes[v2]
                for new_pos_in_v2 in range(len(v2_route)+1):
                    if v2==v1 and new_pos_in_v2==ci_v1:
                        continue
                    d_cost = compute_shifting_dcost(solution, v1, v2, ci_v1, new_pos_in_v2)
                    if d_cost >= -1e-9:
                        continue
                    if is_shifting_potential(solution, v1, v2, ci_v1):
                        potential_args.append(CustomerShiftArgs(d_cost, v1, v2, ci_v1, new_pos_in_v2))
        return potential_args

    def __call__(self, solution: Solution, args: CustomerShiftArgs):
        return self.do(solution, args.v1, args.v2, args.ci_v1, args.new_pos_in_v2)

    # @staticmethod
    # def do_same_route_nb()

    @profile
    def do_same_route(self,
                      original_solution: Solution,
                      v1: int,
                      ci: int,
                      new_pos: int)->Tuple[Solution, bool]:
        solution = original_solution.copy()
        original_route = solution.routes[v1]
        cust_idx = original_route[ci]
        new_route = original_route.copy()
        new_route = new_route[:ci] + new_route[ci+1:]
        new_route = new_route[:new_pos] + [cust_idx] + new_route[new_pos:]
        solution, is_new_route_applicable = apply_new_route(solution, v1, new_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True

    @profile
    def do(self, original_solution:Solution, v1: int, v2: int, ci_v1: int, new_pos_in_v2: int)->Tuple[Solution, bool]:
        if v1==v2:
            return self.do_same_route(original_solution, v1, ci_v1, new_pos_in_v2)
        solution = original_solution.copy()
        cust_idx = original_solution.routes[v1][ci_v1]
        
        v1_route = original_solution.routes[v1]
        new_v1_route = v1_route.copy()
        del new_v1_route[ci_v1]
        new_v2_route = original_solution.routes[v2].copy()
        new_v2_route = new_v2_route[:new_pos_in_v2] + [cust_idx] + new_v2_route[new_pos_in_v2:]

        solution, is_new_route_applicable = apply_new_route(solution, v1, new_v1_route)
        if not is_new_route_applicable:
            return original_solution, False
        solution, is_new_route_applicable = apply_new_route(solution, v2, new_v2_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True
    
    def __repr__(self):
        return "customer-shift"
    
@dataclass
class RouteInterchangeArgs(LocalSearchArgs):
    start_idx: int
    end_idx: int
    
@profile
def compute_same_route_interchange_d_cost(solution: Solution,
                                          v1:int,
                                          start_idx:int,
                                          end_idx:int):
    original_route = solution.routes[v1]
    new_route = original_route.copy()
    new_route[start_idx:end_idx+1] = new_route[start_idx:end_idx+1][::-1]
    new_cost = solution.problem.compute_route_total_distance(new_route)*solution.vehicle_variable_costs[v1]
    original_cost = solution.problem.compute_route_total_distance(original_route)*solution.vehicle_variable_costs[v1]
    d_cost = new_cost-original_cost
    return d_cost

@profile
def compute_route_interchange_d_cost(solution: Solution,
                                          v1:int,
                                          v2:int,
                                          start_idx:int,
                                          end_idx:int):
    problem = solution.problem
    original_v1_route = solution.routes[v1]
    original_v2_route = solution.routes[v2]
    combined_route = solution.routes[v1] + solution.routes[v2]
    combined_route[start_idx:end_idx+1] = combined_route[start_idx:end_idx+1][::-1]
    new_v1_route = combined_route[:len(original_v1_route)]
    new_v2_route = combined_route[len(original_v1_route):]
    d_distance_v1 = problem.compute_route_total_distance(new_v1_route)-problem.compute_route_total_distance(original_v1_route)
    d_distance_v2 = problem.compute_route_total_distance(new_v2_route)-problem.compute_route_total_distance(original_v2_route)
    d_cost = d_distance_v1*solution.vehicle_variable_costs[v1] + d_distance_v2*solution.vehicle_variable_costs[v2]
    return d_cost

@profile
def is_interchange_potential(solution: Solution,
                             v1: int,
                             v2: int,
                             start_idx: int,
                             end_idx: int):
    problem = solution.problem
    if problem.vehicle_reefer_flags[v1] and not problem.vehicle_reefer_flags[v2]:
        return False
    if problem.vehicle_reefer_flags[v2] and not problem.vehicle_reefer_flags[v1]:
        return False
    
    
    original_v1_route = solution.routes[v1]
    combined_route = solution.routes[v1] + solution.routes[v2]
    combined_route[start_idx:end_idx+1] = combined_route[start_idx:end_idx+1][::-1]
    new_v1_route = combined_route[:len(original_v1_route)]
    new_v2_route = combined_route[len(original_v1_route):]
    
    new_filled_volumes_v1 = np.sum(problem.total_demand_volumes[new_v1_route])
    new_filled_volumes_v2 = np.sum(problem.total_demand_volumes[new_v2_route])
    if new_filled_volumes_v1 > problem.vehicle_volume_capacities[v1]:
        return False
    if new_filled_volumes_v2 > problem.vehicle_volume_capacities[v2]:
        return False
    new_filled_weights_v1 = np.sum(problem.total_demand_weights[new_v1_route])
    new_filled_weights_v2 = np.sum(problem.total_demand_weights[new_v2_route])
    if new_filled_weights_v1 > problem.vehicle_weight_capacities[v1]:
        return False
    if new_filled_weights_v2 > problem.vehicle_weight_capacities[v2]:
        return False
    return True


class RouteInterchange(LocalSearchOperator):
    def get_all_potential_args(self, solution:Solution)->List[RouteInterchangeArgs]:
        potential_args:List[RouteInterchangeArgs] = []
        problem = solution.problem
        non_empty_routes_idx = [vi for vi in range(problem.num_vehicles) if len(solution.routes[vi])>0]
        for v1 in range(problem.num_vehicles):
            if not v1 in non_empty_routes_idx:
                continue
            total_length = len(solution.routes[v1])
            for v2 in range(v1, problem.num_vehicles):
                if not v2 in non_empty_routes_idx:
                    continue
                if v1==v2 and len(solution.routes[v1])==1:
                    continue
                if v1 != v2:
                    total_length += len(solution.routes[v2])
                pairs = combinations(range(total_length), 2)
                for start_idx, end_idx in pairs:
                    if v1==v2:
                        if end_idx <= start_idx:
                            continue
                        d_cost = compute_same_route_interchange_d_cost(solution, v1, start_idx, end_idx)
                    else:
                        d_cost = compute_route_interchange_d_cost(solution, v1, v2, start_idx, end_idx)
                    if d_cost>=-1e-9:
                        continue
                    if not is_interchange_potential(solution, v1, v2, start_idx, end_idx):
                        continue
                    args = RouteInterchangeArgs(d_cost, v1, v2, start_idx, end_idx)
                    potential_args.append(args)
        return potential_args
    
    def __call__(self, solution:Solution, args:RouteInterchangeArgs):
        return self.do(solution, args.v1, args.v2, args.start_idx, args.end_idx)
        
    @profile
    def do_same_route(self, original_solution:Solution, v1:int, start_idx:int, end_idx:int)->Tuple[Solution, bool]:
        original_route = original_solution.routes[v1]
        solution = original_solution.copy()
        new_route = original_route.copy()
        new_route[start_idx:end_idx+1] = new_route[start_idx:end_idx+1][::-1]
        solution, is_new_route_applicable = apply_new_route(solution, v1, new_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True
    
    @profile
    def do(self, original_solution:Solution, v1:int, v2:int, start_idx:int, end_idx:int)->Tuple[Solution, bool]:
        if v1 == v2:
            return self.do_same_route(original_solution, v1, start_idx, end_idx)
        solution = original_solution.copy()
        original_v1_route = solution.routes[v1]
        combined_route = solution.routes[v1] + solution.routes[v2]
        combined_route[start_idx:end_idx+1] = combined_route[start_idx:end_idx+1][::-1]
        new_v1_route = combined_route[:len(original_v1_route)]
        new_v2_route = combined_route[len(original_v1_route):]
        solution, is_new_route_applicable = apply_new_route(solution, v1, new_v1_route)
        if not is_new_route_applicable:
            return original_solution, False
        solution, is_new_route_applicable = apply_new_route(solution, v2, new_v2_route)
        if not is_new_route_applicable:
            return original_solution, False
        return solution, True
    
    def __repr__(self):
        return "route-interchange"
        