import math
from typing import List

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from line_profiler import profile
from problem.hvrp3l import HVRP3L
from problem.solution import NO_VEHICLE, Solution
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo_interface.arr2 import RepairMechanism


def get_routes_from_x(x: np.ndarray, problem: HVRP3L)->List[List[int]]:
    """
        not necessarily feasible, hopefully (should!) x has been repaired
    """
    routes = [[] for _ in range(problem.num_vehicles)] # first dim for vehicle, second for orderings

    for i in range(problem.num_customers, 2*problem.num_customers):
        ci = i-problem.num_customers
        cust_idx = problem.customers[ci].idx
        if problem.node_reefer_flags[cust_idx]:
            vi = math.floor(x[i]*problem.num_reefer_trucks)
        else:
            vi = math.floor(x[i]*problem.num_vehicles)
        routes[vi].append(cust_idx)

    for vi in range(problem.num_vehicles):
        priorities = np.asanyarray([x[cust_idx-1] for cust_idx in routes[vi]])
        routes[vi] = np.asanyarray(routes[vi])
        sorted_idx = np.argsort(priorities)
        routes[vi] = routes[vi][sorted_idx]        

    return routes

def hash_x_to_ndarray(x: np.ndarray, problem: HVRP3L)->np.ndarray:
    """
        hash? an individu for duplicate elimination
    """
    priorities = np.empty((problem.num_customers,), dtype=float)
    arr = np.empty((2, problem.num_customers), dtype=int) # first dim for vehicle, second for orderings

    for i in range(problem.num_customers, 2*problem.num_customers):
        ci = i-problem.num_customers
        cust_idx = problem.customers[ci].idx
        if problem.node_reefer_flags[cust_idx]:
            vi = math.floor(x[i]*problem.num_reefer_trucks)
        else:
            vi = math.floor(x[i]*problem.num_vehicles)
        arr[0, ci] = vi

    for i in range(problem.num_customers):
        priorities[i] = x[i]
    arr[1, :] = np.argsort(priorities)
    return arr



class DuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, problem: HVRP3L, **kwargs):
        super().__init__(**kwargs)
        self.problem: HVRP3L = problem

    def is_equal(self, a:Individual, b:Individual):
        arr_a = hash_x_to_ndarray(a.x, self.problem)
        arr_b = hash_x_to_ndarray(b.x, self.problem)
        return np.array_equal(arr_a, arr_b)
    

class HVRP3L_OPT(ElementwiseProblem):
    def __init__(self,
                 hvrp3l_instance: HVRP3L,
                 repair_mechanism: RepairMechanism,
                 **kwargs):
        super().__init__(elementwise=True, **kwargs)
        self.hvrp3l_instance = hvrp3l_instance
        self.n_var = 2*hvrp3l_instance.num_customers
        self.add_remaining_requests = repair_mechanism.repair

        # first num_customers dims are for customer priority
        # the next num_customers dims are for vehicle assignment
        self.xl = np.zeros([self.n_var, ], dtype=float)
        self.xu = np.ones([self.n_var, ], dtype=float)

    def decode(self, x: np.ndarray)->Solution:
        problem: HVRP3L = self.hvrp3l_instance
        solution: Solution = Solution(problem)
        customers = solution.problem.customers
        # try to map first into vehicle
        # if not feasible?
        
        for i in range(problem.num_customers, 2*problem.num_customers):
            ci = i-problem.num_customers
            cust_idx = customers[ci].idx
            if solution.node_reefer_flags[cust_idx]:
                vi = math.floor(x[i]*problem.num_reefer_trucks)
            else:
                vi = math.floor(x[i]*problem.num_vehicles)
            solution.node_vhc_assignment_map[cust_idx] = vi

        for vi in range(problem.num_vehicles):
            vi_node_idx = np.nonzero(solution.node_vhc_assignment_map==vi)[0]
            vi_x_idx = vi_node_idx-1 # it's to get their idx in x (chromosome/individu,i.e., cust 2 is in x[1])
            if len(vi_node_idx) == 0:
                continue

            # simple capacity checking
            total_weights = np.sum(solution.node_demand_weights[vi_node_idx])
            if total_weights>problem.vehicle_weight_capacities[vi]:
                continue
            total_volumes = np.sum(solution.node_demand_volumes[vi_node_idx])
            if total_volumes>problem.vehicle_volume_capacities[vi]:
                continue

            
            cust_priorities = x[vi_x_idx]
            sorted_idx = np.argsort(cust_priorities)
            vi_node_idx = vi_node_idx[sorted_idx]

            total_num_items = sum(solution.node_num_items[cust_idx] for cust_idx in vi_node_idx)
            
            # this all actually can be pre-allocated in the problem interface
            # and used freely, to remove allocation time
            item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
            item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            n = 0
            for i, cust_idx in enumerate(vi_node_idx):
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
            positions, rotations, is_packing_feasible = packing_result
            
            if not is_packing_feasible:
                solution.node_vhc_assignment_map[vi_node_idx] = NO_VEHICLE
                continue
            # now commit the route, because we can pack
            solution.node_vhc_assignment_map[vi_node_idx] = vi
            solution.filled_volumes[vi] = total_volumes
            solution.filled_weight_caps[vi] = total_weights
            solution.routes[vi] += vi_node_idx.tolist()
            n = 0
            for i, cust_idx in enumerate(vi_node_idx):
                c_num_items = solution.node_num_items[cust_idx]
                item_mask = problem.node_item_mask[cust_idx, :]
                solution.item_positions[item_mask] = positions[n:n+c_num_items]
                solution.item_rotations[item_mask] = rotations[n:n+c_num_items]
                n += c_num_items

        for vi in range(solution.num_vehicles):
            if len(solution.routes[vi]) == 0:
                continue
            solution.total_vehicle_fixed_cost += solution.vehicle_fixed_costs[vi]
            total_distance = solution.problem.compute_route_total_distance(solution.routes[vi])
            solution.total_vehicle_variable_cost += total_distance*solution.vehicle_variable_costs[vi]
        solution = self.add_remaining_requests(solution)
        return solution
    
    def get_total_cost_without_decoding(self, x:np.ndarray)->float:
        routes = get_routes_from_x(x, self.hvrp3l_instance)
        total_fixed_cost = 0
        total_variable_cost = 0
        for vi, route in enumerate(routes):
            if len(route)==0:
                continue
            total_fixed_cost += self.hvrp3l_instance.vehicle_fixed_costs[vi]
            distance = self.hvrp3l_instance.compute_route_total_distance(route)
            total_variable_cost += distance*self.hvrp3l_instance.vehicle_variable_costs[vi]
        return total_fixed_cost + total_variable_cost


    # this is the decoding method
    @profile
    def _evaluate(self, x:np.ndarray, out:dict, *args, **kwargs):
        total_cost = self.get_total_cost_without_decoding(x)
        out["F"] = total_cost
            

class RepairEncoding(Repair):
    def _do(self, problem: HVRP3L_OPT, X:np.ndarray, **kwargs)->np.ndarray:
        hvrp3l_instance = problem.hvrp3l_instance
        for i, x in enumerate(X):
            solution = problem.decode(x)
            for vi, route in enumerate(solution.routes):
                for cust_idx in route:
                    vxi = hvrp3l_instance.num_customers + cust_idx-1
                    if hvrp3l_instance.node_reefer_flags[cust_idx]:
                        vi_original = math.floor(x[vxi]*hvrp3l_instance.num_reefer_trucks)
                    else:
                        vi_original = math.floor(x[vxi]*hvrp3l_instance.num_vehicles)
                    if vi_original==vi:
                        continue
                    
                    if hvrp3l_instance.node_reefer_flags[cust_idx]:
                        new_x_vi = float(vi)/hvrp3l_instance.num_reefer_trucks + 1/(2*hvrp3l_instance.num_reefer_trucks)
                    else:
                        new_x_vi = float(vi)/hvrp3l_instance.num_vehicles + 1/(2*hvrp3l_instance.num_vehicles)
                    x[vxi] = new_x_vi
                cis = [cust_idx-1 for cust_idx in route]
                priorities = x[cis]
                sorted_priorites = np.sort(priorities)
                x[cis] = sorted_priorites
                X[i] = x
        return X
