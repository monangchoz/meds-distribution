import heapq
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numba as nb
import numpy as np
from avns.utils import apply_new_route, try_packing_custs_in_route
from ep_heuristic.insertion import argsort_items
from ep_heuristic.random_slpack import try_slpack
from line_profiler import profile
from problem.hvrp3l import HVRP3L
from problem.item import POSSIBLE_ROTATION_PERMUTATION_MATS
from problem.solution import NO_VEHICLE, Solution


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



@nb.jit(nb.types.Tuple((nb.float64[:,:], nb.int64[:,:], nb.int64, nb.int64, nb.bool))(nb.int64[:],nb.int64[:],nb.float64[:,:,:],nb.int64[:,:],nb.int64[:,:],nb.int64[:],nb.float64[:,:],nb.int64[:,:]), parallel=True)
def find_packable_insertion_point_nb(vehicles_idx_arr: np.ndarray,
                                     positions_arr: np.ndarray,
                                     all_item_dims: np.ndarray,
                                     all_item_priorities: np.ndarray,
                                     all_item_sorted_idxs: np.ndarray,
                                     total_num_items_list: np.ndarray,
                                     container_dims: np.ndarray,
                                     possible_rotation_permutation_mats: np.ndarray)->Tuple[np.ndarray, np.ndarray, int, int, bool]:
    
    num_possible_positions = positions_arr.shape[0]
    batch_size = 4
    num_batches = math.ceil(num_possible_positions/batch_size)
    
    max_total_num_items = np.max(total_num_items_list)
    all_rotation_trial_idxs = np.zeros((batch_size, max_total_num_items, 2), dtype=np.int64)
    all_rotation_trial_idxs[:,:,1] = 1

    batch_item_positions = np.empty((batch_size, max_total_num_items, 3), dtype=np.float64)
    batch_item_rotations = np.empty((batch_size, max_total_num_items, 3), dtype=np.int64)
    batch_feasibilities  = np.empty((batch_size, ), dtype=np.bool_)
    for n in range(num_batches):
        for m in nb.prange(batch_size):
            i = n*batch_size + m
            if i >= num_possible_positions:
                continue
            container_dim = container_dims[i]
            total_num_items = total_num_items_list[i]
            item_dims = all_item_dims[i,:total_num_items]
            item_priorities = all_item_priorities[i, :total_num_items]
            sorted_idx = all_item_sorted_idxs[i, :total_num_items]
            all_rotation_trial_idxs[m,:total_num_items,0]=0
            all_rotation_trial_idxs[m,:total_num_items,1]=1
            rotation_trial_idx = all_rotation_trial_idxs[m, :total_num_items]
            batch_item_positions[m, :total_num_items], batch_item_rotations[m, :total_num_items], batch_feasibilities[m] = try_slpack(item_dims, item_priorities, sorted_idx, rotation_trial_idx, container_dim, possible_rotation_permutation_mats, 0.8, 5)
        for m in range(batch_size):
            i = n*batch_size + m
            if i >= num_possible_positions:
                continue
            if batch_feasibilities[m]:
                total_num_items = total_num_items_list[i]
                return batch_item_positions[m, :total_num_items], batch_item_rotations[m, :total_num_items], vehicles_idx_arr[i], positions_arr[i], True

    return np.empty((1,1),dtype=np.float64), np.empty((1,1), dtype=np.int64), 0, 0, False
    # item_positions, item_rotations, vi, pos, is_any_vi_pos_feasible  

@profile
def find_packable_insertion_point(solution: Solution, cust_idx: int, vehicles_idx:List[int], positions:List[int])->Tuple[np.ndarray, np.ndarray, int, int, bool]:
    problem = solution.problem
    num_all_items = np.sum(solution.node_num_items)
    num_possible_pos = len(positions)
    
    all_item_dims: np.ndarray = np.empty([num_possible_pos, num_all_items, 3], dtype=float)
    all_item_volumes: np.ndarray = np.empty([num_possible_pos, num_all_items], dtype=float)
    all_item_weights: np.ndarray = np.empty([num_possible_pos, num_all_items], dtype=float)
    all_item_priorities: np.ndarray = np.empty([num_possible_pos, num_all_items], dtype=int)
    all_item_sorted_idxs: np.ndarray = np.empty([num_possible_pos, num_all_items], dtype=int)
    total_num_items_list: np.ndarray = np.empty((num_possible_pos,), dtype=int)
    vehicles_idx_arr: np.ndarray = np.asanyarray(vehicles_idx, dtype=int)
    positions_arr: np.ndarray = np.asanyarray(positions, dtype=int)
    container_dims: np.ndarray = np.empty([num_possible_pos, 3], dtype=float)
    for bi, (vi, pos) in enumerate(zip(vehicles_idx, positions)):
        old_route = solution.routes[vi].copy()
        new_route = old_route[:pos] + [cust_idx] + old_route[pos:]
        n = 0
        total_num_items = np.sum(solution.node_num_items[new_route])
        total_num_items_list[bi] = total_num_items
        container_dims[bi] = problem.vehicle_container_dims[vi]
        for i, vi_cust_idx in enumerate(new_route):
            c_num_items = solution.node_num_items[vi_cust_idx]
            item_mask = problem.node_item_mask[vi_cust_idx, :]
            all_item_dims[bi, n:n+c_num_items] = problem.item_dims[item_mask]
            all_item_volumes[bi, n:n+c_num_items] = problem.item_volumes[item_mask]
            all_item_weights[bi, n:n+c_num_items] = problem.item_weights[item_mask]
            all_item_priorities[bi, n:n+c_num_items] = i
            n += c_num_items
        item_base_areas = all_item_dims[bi, :total_num_items, 0]*all_item_dims[bi, :total_num_items, 1]
        all_item_sorted_idxs[bi,:total_num_items] = argsort_items(item_base_areas, all_item_volumes[bi, :total_num_items], all_item_priorities[bi, :total_num_items])

    item_positions, item_rotations, vi, pos, is_any_vi_pos_feasible = find_packable_insertion_point_nb(vehicles_idx_arr, positions_arr, all_item_dims, all_item_priorities,  all_item_sorted_idxs, total_num_items_list, container_dims, POSSIBLE_ROTATION_PERMUTATION_MATS)
    return item_positions, item_rotations, vi, pos, is_any_vi_pos_feasible



class Diversification:
    def __init__(self, num_nodes: int):
        self.non_imp: int = 0
        self.operators: List[Callable] = [self.ruin_reconstruct]
        self.imp_number: np.ndarray = np.zeros((len(self.operators),), dtype=int)
        self.call_number: np.ndarray = np.zeros((len(self.operators),), dtype=int)
        self.best_cost: Optional[float] = None
        self.last_op_idx: int = 999999
        self.edge_eliminated_counts: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=int)
        
    def update_improvement_status(self, current_solution: Solution):
        if self.best_cost is None:
            self.best_cost = current_solution.total_cost
            return
        if self.best_cost >= current_solution.total_cost:
            self.non_imp += 1
        else:
            self.imp_number[self.last_op_idx] += 1
            self.non_imp = 0
        
    def __call__(self, solution: Solution)->Solution:
        score = (1+self.imp_number)/(1+self.call_number)
        probs = score/np.sum(score)
        chosen_operator_idx = np.random.choice(len(self.operators), size=1, p=probs).item()
        self.last_op_idx = chosen_operator_idx
        self.call_number[chosen_operator_idx] += 1
        chosen_operator = self.operators[chosen_operator_idx]
        new_solution = chosen_operator(solution)
        return new_solution

    @profile
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
            solution.total_vehicle_variable_cost -= problem.compute_route_total_distance(route)*problem.vehicle_variable_costs[vi]
            solution.total_vehicle_fixed_cost -= solution.vehicle_fixed_costs[vi]
            solution.routes[vi] = []
            
        # sort unvisited customers based on volume and weight
        unvisited_custs_idx = np.where(solution.node_vhc_assignment_map==NO_VEHICLE)[0]
        demand_volumes = solution.node_demand_volumes[unvisited_custs_idx]
        sorted_idx = np.argsort(-demand_volumes)
        unvisited_custs_idx = unvisited_custs_idx[sorted_idx]
        unvisited_custs_idx = unvisited_custs_idx.tolist()
        # try to insert
        for cust_idx in unvisited_custs_idx:
            # print("hello", cust_idx, flush=True)
            vehicles_idx, positions, d_costs = get_possible_insertion_positions(solution, cust_idx)
            item_positions, item_rotations, vi, pos, is_any_vi_pos_feasible = find_packable_insertion_point(solution, cust_idx, vehicles_idx, positions)
            if not is_any_vi_pos_feasible:
                return original_solution
            old_route = solution.routes[vi].copy()
            new_route = old_route[:pos] + [cust_idx] + old_route[pos:]
            solution, _ = apply_new_route(solution, vi, new_route, item_positions, item_rotations)
        return solution