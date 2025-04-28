from dataclasses import dataclass
import math
import random
from typing import Tuple, List, Callable, Optional

import numpy as np
from problem.hvrp3l import HVRP3L
from problem.solution import Solution, NO_VEHICLE
from avns.utils import apply_new_route, try_packing_custs_in_route

def get_possible_insertion_positions(solution: Solution, cust_idx:int)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    problem = solution.problem
    positions = []
    vehicles_idx = []
    d_costs = []
    distance_matrix = problem.distance_matrix
    for vi in range(solution.num_vehicles):
        if solution.node_reefer_flags[cust_idx] and not solution.vehicle_reefer_flags[vi]:
            continue
        if solution.filled_volumes[vi] + solution.node_demand_volumes[cust_idx] > solution.vehicle_volume_capacities[vi]:
            continue
        if solution.filled_weight_caps[vi] + solution.node_demand_weights[cust_idx] > solution.vehicle_weight_capacities[vi]:
            continue
        route = solution.routes[vi]
        route_len = len(route)
        d_cost = 0
        for ci in range(route_len+1):
            if route_len==0:
                d_cost += solution.vehicle_fixed_costs[vi]
            prev_node = 0
            if ci>0:
                prev_node = route[ci-1]
            next_node = 0
            if ci + 1 < route_len:
                next_node = route[ci+1]
            d_distance = distance_matrix[prev_node, cust_idx] + distance_matrix[cust_idx, next_node] - distance_matrix[prev_node,next_node]
            d_cost += d_distance*solution.vehicle_variable_costs[vi]
            positions.append(ci)
            vehicles_idx.append(vi)
            d_costs.append(d_cost)
    positions_ = np.asanyarray(positions)
    vehicles_idx_ = np.asanyarray(vehicles_idx)
    d_costs_ = np.asanyarray(d_costs)
    sorted_idx = np.argsort(d_costs_)
    positions_ = positions_[sorted_idx]
    vehicles_idx_ = vehicles_idx_[sorted_idx] 
    d_costs_ = d_costs_[sorted_idx]
    return vehicles_idx_, positions_, d_costs_


@dataclass
class SplitEdge:
    cost: float
    vehicle_idx: int

def generate_split_edge_matrix(solution: Solution, giant_route:List[int], is_reefer:bool)->List[List[Optional[SplitEdge]]]:
    # generate graph
    problem = solution.problem
    num_nodes = solution.problem.num_nodes
    split_edge_matrix: List[List[Optional[SplitEdge]]] = [[None]*num_nodes]*num_nodes
    start_idx = 0
    # representative
    vi = problem.num_reefer_trucks
    if is_reefer:
        vi = 0
    node_in_gr_reefer_flags = problem.node_reefer_flags[giant_route]
    node_in_gr_demand_weights = problem.total_demand_weights[giant_route]
    node_in_gr_demand_volumes = problem.total_demand_volumes[giant_route]
    while start_idx < num_nodes-1:
        for end_idx in range(num_nodes, start_idx, -1):
            if np.any(node_in_gr_reefer_flags[start_idx:end_idx]) and not problem.vehicle_reefer_flags[vi]:
                continue
            
            total_weight = node_in_gr_demand_weights[start_idx:end_idx]
            total_volume = node_in_gr_demand_volumes[start_idx:end_idx]
            is_weight_capacity_enough = total_weight<=problem.vehicle_weight_capacities[vi]
            is_volume_capacity_enough = total_volume<=problem.vehicle_volume_capacities[vi]
            if not is_weight_capacity_enough or not is_volume_capacity_enough:
                continue
            _,_, is_packing_feasible = try_packing_custs_in_route(solution, vi, giant_route[start_idx:end_idx])
            if not is_packing_feasible:
                continue
            
            for i in range(start_idx, end_idx):
                total_cost = problem.vehicle_fixed_costs[vi]
                ci = giant_route[start_idx]
                for j in range(i+1, end_idx):
                    cj = giant_route[j]
                    distance = problem.distance_matrix[ci, cj]
                    cost = distance*problem.vehicle_variable_costs[vi]
                    total_cost += cost
                    edge = SplitEdge(total_cost, vi)
                    split_edge_matrix[ci][cj] = edge
    return split_edge_matrix
            


class Diversification:
    def __init__(self, problem: HVRP3L):
        self.non_imp: int = 0
        self.operators: List[Callable] = [self.ruin_reconstruct]
        self.imp_number: np.ndarray = np.zeros((len(self.operators),), dtype=int)
        self.call_number: np.ndarray = np.zeros((len(self.operators),), dtype=int)
        self.best_cost: Optional[float] = None
        self.last_op_idx: int = 999999
        self.edge_eliminated_counts: np.ndarray = np.zeros((problem.num_nodes, problem.num_nodes), dtype=int)
        
    def update_improvement_status(self, current_solution: Solution):
        if self.best_cost is None:
            self.best_cost = current_solution.total_cost
            return
        if self.best_cost >= current_solution.total_cost:
            self.non_imp += 1
        else:
            self.imp_number[self.last_op_idx] += 1
        
    def __call__(self, solution: Solution)->Solution:
        score = (1+self.imp_number)/(1+self.call_number)
        probs = score/np.sum(score)
        chosen_operator_idx = np.random.choice(2, size=1, p=probs).item()
        self.last_op_idx = chosen_operator_idx
        self.call_number[chosen_operator_idx] += 1
        chosen_operator = self.operators[chosen_operator_idx]
        new_solution = chosen_operator(solution)
        return new_solution

    def ruin_reconstruct(self, original_solution: Solution)->Solution:
        solution = original_solution.copy()
        problem = solution.problem
        # select vehicles to deconstruct
        non_empty_vehicles_idx = [vi for vi in range(problem.num_vehicles) if len(solution.routes[vi])>0]
        num_chosen_vehicles = random.randint(1, len(non_empty_vehicles_idx))
        chosen_vehicles_idx = random.sample(non_empty_vehicles_idx, k=num_chosen_vehicles)
        
        for vi in chosen_vehicles_idx:
            route = solution.routes[vi]
            solution.node_vhc_assignment_map[route] = NO_VEHICLE
            solution.filled_volumes[vi] -= np.sum(solution.node_demand_volumes[route])
            solution.filled_weight_caps[vi] -= np.sum(solution.node_demand_weights[route])
            solution.vehicle_variable_costs -= problem.compute_route_total_distance(route)*problem.vehicle_variable_costs[vi]
            solution.vehicle_fixed_costs -= solution.vehicle_fixed_costs[vi]

        # sort unvisited customers based on volume and weight
        unvisited_custs_idx = np.where(solution.node_vhc_assignment_map==NO_VEHICLE)[0]
        demand_volumes = solution.node_demand_volumes[unvisited_custs_idx]
        sorted_idx = np.argsort(-demand_volumes)
        unvisited_custs_idx = unvisited_custs_idx[sorted_idx]              
        
        # try to insert
        for cust_idx in unvisited_custs_idx:
            vehicles_idx, positions, d_costs = get_possible_insertion_positions(solution, cust_idx)
            cust_insertion_feasible: bool = False
            for vi, pos in zip(vehicles_idx, positions):
                old_route = solution.routes[vi].copy()
                new_route = old_route[:pos] + [cust_idx] + old_route[pos:]
                solution, is_insertion_possible = apply_new_route(solution, vi, new_route)
                if is_insertion_possible:
                    cust_insertion_feasible = True
                    break
            if not cust_insertion_feasible:
                return original_solution

        return solution
    
    def concat(self, solution: Solution)->List[int]:
        distance_matrix = solution.problem.distance_matrix
        non_empty_routes_idx = [vi for vi in range(solution.num_vehicles) if len(solution.routes[vi])>0]
        first_vi = random.choice(non_empty_routes_idx)
        non_empty_routes_idx.remove(first_vi)
        giant_route = solution.routes[first_vi].copy()
        while len(non_empty_routes_idx) > 0:
            starting_nodes = [solution.routes[vi][0] for vi in non_empty_routes_idx]
            giant_route_last_node = giant_route[-1]
            distances_to_giant_route = distance_matrix[giant_route_last_node][starting_nodes]
            sorted_idx = np.argsort(distances_to_giant_route)
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(len(ranks))
            m = len(non_empty_routes_idx)
            probs = 2*(m-ranks+1)/(m*(m+1))
            selected_vi = np.random.choice(non_empty_routes_idx, size=1, p=probs).item()
            giant_route += solution.routes[selected_vi]
            non_empty_routes_idx.remove(first_vi)
        return giant_route
            
    def alter(self, solution: Solution, giant_route):
        distance_matrix = solution.problem.distance_matrix
        dij = distance_matrix[giant_route[:-1], giant_route[1:]]
        avgi = np.average(distance_matrix[giant_route[:-1]], axis=1)
        avgj = np.average(distance_matrix[giant_route[1:]], axis=1)
        avgij = (avgi+avgj)/2
        etij = self.edge_eliminated_counts[giant_route[:-1], giant_route[1:]]
        uij = (dij/avgij)/(1+etij)
        
        i = np.argmax(uij)
        j = i+1
        giant_route = giant_route[j:]+giant_route[:i+1]
        
        num_pairs_to_swap = math.trunc(self.non_imp/2)
        for _ in range(num_pairs_to_swap):
            i, j = np.random.choice(len(giant_route), size=2, replace=False)
            tmp = giant_route[i]
            giant_route[i] = giant_route[j]
            giant_route[j] = tmp
        return giant_route
        
    def split(self, solution:Solution, giant_route:List[int])->Solution:
        problem = solution.problem
        num_nodes = solution.problem.num_nodes
        split_multi_edge_matrix: List[List[List[SplitEdge]]] = [[[]]*num_nodes]*num_nodes
        reefer_split_edge_matrix = generate_split_edge_matrix(solution, giant_route, is_reefer=True)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if reefer_split_edge_matrix[i][j] is None:
                    continue
                edge = reefer_split_edge_matrix[i][j]
                # duplicate for every possible reefer matrix
                for vi in range(problem.num_reefer_trucks):
                    new_edge = SplitEdge(edge.cost, vi)
                    split_multi_edge_matrix[i][j].append(new_edge)
        
        
        normal_split_edge_matrix = generate_split_edge_matrix(solution, giant_route, is_reefer=False)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if normal_split_edge_matrix[i][j] is None:
                    continue
                edge = normal_split_edge_matrix[i][j]
                # duplicate for every possible reefer matrix
                for vi in range(problem.num_reefer_trucks, problem.num_vehicles):
                    new_edge = SplitEdge(edge.cost, vi)
                    split_multi_edge_matrix[i][j].append(new_edge)
        
            
            
        # solve dijkstra on (node, bitmask)
        # convert shortest path route into routes and loading plans
        # return new solution
        return solution
    
    def concat_split(self, original_solution: Solution)->Solution:
        giant_route = self.concat(original_solution)
        giant_route = self.alter()
        # final_routes_with_info, is_splitting_feasible = self.split(original_solution, giant_route)
        solution = original_solution.copy()
        # if is_splitting_feasible:
        #     # apply final routes
        return solution 