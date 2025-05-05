from typing import Tuple

import numpy as np

from problem.solution import Solution, NO_VEHICLE
from problem.hvrp3l import HVRP3L
from avns.utils import apply_new_route

def combine_route(solution: Solution, vi: int, vj: int, vt:int)->Tuple[Solution: bool]:
    new_solution = solution.copy()
    problem = solution.problem
    new_route = solution.routes[vi] + solution.routes[vj]
    new_solution.node_vhc_assignment_map[solution.routes[vi]] = NO_VEHICLE
    new_solution.node_vhc_assignment_map[solution.routes[vj]] = NO_VEHICLE
    new_solution.filled_volumes[vi] = 0
    new_solution.filled_volumes[vj] = 0
    new_solution.filled_weight_caps[vi] = 0
    new_solution.filled_weight_caps[vj] = 0
    d_vcost = - (problem.compute_route_total_distance(solution.routes[vi])*problem.vehicle_variable_costs[vi] + problem.compute_route_total_distance(solution.routes[vj])*problem.vehicle_variable_costs[vj])
    d_fcost = - (problem.vehicle_fixed_costs[vi] + problem.vehicle_fixed_costs[vj])
    # print(solution.total_vehicle_fixed_cost, d_fcost, problem.vehicle_fixed_costs[vt])
    new_solution.total_vehicle_fixed_cost += d_fcost
    new_solution.total_vehicle_variable_cost += d_vcost
    new_solution.routes[vi]=[]
    new_solution.routes[vj]=[]
    # print(solution.routes[vi], solution.routes[vj], solution.routes[vt])
    new_solution, is_combination_packable = apply_new_route(new_solution, vt, new_route)
    if not is_combination_packable:
        return solution, False
    # print(new_solution.routes[vi], new_solution.routes[vj], new_solution.routes[vt])
    return new_solution, True

def saving(problem: HVRP3L)->Solution:
    solution = Solution(problem)
    for cust_idx in range(1, problem.num_nodes):
        for v_idx in range(problem.num_vehicles):
            if len(solution.routes[v_idx])>0:
                continue
            
            if problem.node_reefer_flags[cust_idx] and not solution.vehicle_reefer_flags[v_idx]:
                continue
            
            if not problem.node_reefer_flags[cust_idx] and solution.vehicle_reefer_flags[v_idx]:
                continue
            
            solution, is_feasible = apply_new_route(solution, v_idx, [cust_idx])
            if not is_feasible:
                raise ValueError()
            break
        
    possible_combination_exists = True
    while possible_combination_exists:
        # print(solution.total_cost)
        possible_combination_exists = False
        vi_vj_vt_dcost_list = []
        for vi in range(problem.num_vehicles):
            if len(solution.routes[vi])==0:
                continue
            for vj in range(vi+1, problem.num_vehicles):
                if len(solution.routes[vj])==0:
                    continue
                total_volume = solution.filled_volumes[vi] + solution.filled_volumes[vj]
                total_weight = solution.filled_volumes[vi] + solution.filled_volumes[vj]
                combined_nodes = solution.routes[vi] + solution.routes[vj]
                is_need_reefer = np.any(problem.node_reefer_flags[combined_nodes])
                new_route1 = solution.routes[vi]+solution.routes[vj]
                new_route2 = solution.routes[vj]+solution.routes[vi]
                d_vcost = - (problem.compute_route_total_distance(solution.routes[vi])*problem.vehicle_variable_costs[vi] + problem.compute_route_total_distance(solution.routes[vj])*problem.vehicle_variable_costs[vj])
                new_route1_distance = problem.compute_route_total_distance(new_route1)
                new_route2_distance = problem.compute_route_total_distance(new_route2)
                for vt in [vi, vj]:
                    if is_need_reefer and not problem.vehicle_reefer_flags[vt]:
                        continue
                    if total_volume > problem.vehicle_volume_capacities[vt]:
                        continue
                    if total_weight > problem.vehicle_weight_capacities[vt]:
                        continue
                    d_fcost = problem.vehicle_fixed_costs[vt] - (problem.vehicle_fixed_costs[vi]+problem.vehicle_fixed_costs[vj])
                    d_vcost1 = new_route1_distance*problem.vehicle_variable_costs[vt] + d_vcost
                    d_cost1 = d_fcost+d_vcost1
                    if d_cost1<0:
                        vi_vj_vt_dcost_list.append((vi, vj, vt, d_cost1))
                    d_vcost2 = new_route2_distance*problem.vehicle_variable_costs[vt] + d_vcost
                    d_cost2 = d_fcost+d_vcost2
                    if d_cost2<0:                    
                        vi_vj_vt_dcost_list.append((vj, vi, vt, d_cost2)) 

        for (vi, vj, vt, d_cost) in vi_vj_vt_dcost_list:
            solution, is_combine_feasible = combine_route(solution, vi, vj, vt)
            if is_combine_feasible:
                possible_combination_exists = True
                break
                 
        
        # solution.is_feasible
    # exit()
    return solution