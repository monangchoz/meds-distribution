import numpy as np
from typing import List, Tuple


from problem.solution import Solution, NO_VEHICLE
from ep_heuristic.random_slpack import random_slpack

class RepairMechanism:
    def __init__(self, 
                 num_customers: int, 
                 num_vehicles: int):
        num_possible_positions = num_customers + num_vehicles*2 + 10
        self.vehicle_idxs = np.empty((num_possible_positions,), dtype=int)
        self.positions = np.empty((num_possible_positions,), dtype=int)
        self.insertion_costs = np.empty((num_possible_positions,), dtype=float)
        self.tmp_routes = np.empty((num_vehicles, num_customers+1), dtype=int)
        self.tmp_route_lens = np.empty((num_vehicles,), dtype=int)

    def repair(self, solution: Solution):
        raise NotImplementedError


def get_sorted_possible_insertion_positions(vehicle_idxs:np.ndarray,
                                            positions:np.ndarray,
                                            insertion_costs:np.ndarray,
                                            cust_idx:int, 
                                            cust_demand_volume: float,
                                            cust_demand_weight: float,
                                            customer_need_reefer: bool, 
                                            vehicle_filled_volumes: np.ndarray,
                                            vehicle_filled_weights: np.ndarray,
                                            vehicle_volumes: np.ndarray,
                                            vehicle_weight_capacities: np.ndarray,
                                            vehicle_reefer_flags: np.ndarray,
                                            routes: np.ndarray,
                                            route_lens: np.ndarray,
                                            distance_matrix: np.ndarray,
                                            vehicle_fixed_costs: np.ndarray,
                                            vehicle_variable_costs: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_possible_insertions = 0
    for vehicle_idx, route in enumerate(routes):
        vehicle_has_enough_volume = cust_demand_volume + vehicle_filled_volumes[vehicle_idx] <= vehicle_volumes[vehicle_idx]
        vehicle_has_enough_capacity = cust_demand_weight + vehicle_filled_weights[vehicle_idx] <= vehicle_weight_capacities[vehicle_idx]
        if not (vehicle_has_enough_volume and vehicle_has_enough_capacity):
            continue
        if customer_need_reefer and not vehicle_reefer_flags[vehicle_idx]:
            continue
        route_len = route_lens[vehicle_idx]
        for pos_idx in range(route_len+1):
            if pos_idx == 0:
                prev_node = 0
            else:
                prev_node = route[pos_idx-1]
            
            if pos_idx == route_len:
                next_node = 0
            else:
                next_node = route[pos_idx]

            d_distance = distance_matrix[prev_node, cust_idx] + distance_matrix[cust_idx, next_node] - distance_matrix[prev_node, next_node]
            insertion_cost = d_distance*vehicle_variable_costs[vehicle_idx]
            if route_len == 0:
                insertion_cost += vehicle_fixed_costs[vehicle_idx]

            vehicle_idxs[num_possible_insertions] = vehicle_idx
            positions[num_possible_insertions] = pos_idx
            insertion_costs[num_possible_insertions] = insertion_cost
            num_possible_insertions += 1
    
    sorted_idx = np.argsort(insertion_costs[:num_possible_insertions])
    return vehicle_idxs[sorted_idx], positions[sorted_idx], insertion_costs[sorted_idx]

class ARR1(RepairMechanism):
    
    def get_possible_insertions(self, cust_idx:int, solution:Solution)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for vehicle_idx, route in enumerate(solution.routes):
            self.tmp_routes[vehicle_idx, :len(route)] = route
            self.tmp_route_lens[vehicle_idx] = len(route)
        cust_demand_volume = solution.customer_demand_volumes[cust_idx]
        cust_demand_weight = solution.customer_demand_weights[cust_idx]
        cust_need_reefer = solution.customer_reefer_flags[cust_idx]    
        return get_sorted_possible_insertion_positions(self.vehicle_idxs,
                                                       self.positions,
                                                       self.insertion_costs,
                                                       cust_idx,
                                                       cust_demand_volume,
                                                       cust_demand_weight,
                                                       cust_need_reefer,
                                                       solution.filled_volumes,
                                                       solution.filled_weight_caps,
                                                       solution.vehicle_volume_capacities,
                                                       solution.vehicle_weight_capacities,
                                                       solution.vehicle_reefer_flags,
                                                       self.tmp_routes,
                                                       self.tmp_route_lens,
                                                       solution.problem.distance_matrix,
                                                       solution.vehicle_fixed_costs,
                                                       solution.vehicle_variable_costs)
    
    def try_insertion(self, 
                      cust_idx: int, 
                      vehicle_idx: int, 
                      pos_idx: int, 
                      solution: Solution)->Tuple[np.ndarray,np.ndarray,bool]:
        problem = solution.problem
        customers = problem.customers
        new_route = solution.routes[vehicle_idx]
        new_route = new_route[:pos_idx] + [cust_idx] + new_route[pos_idx:]
        total_num_items = sum(customers[ci].num_items for ci in new_route)

        # this all actually can be pre-allocated in the problem interface
        # and used freely, to remove allocation time
        item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
        item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
        item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
        item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=float)
        n = 0
        for i, ci in enumerate(new_route):
            c_num_items = customers[ci].num_items
            item_mask = problem.customer_item_mask[ci, :]
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
        return packing_result

    
    def repair(self, solution: Solution)->Solution:
        unvisited_customer_idxs: np.ndarray = np.where(solution.cust_vhc_assignment_map==NO_VEHICLE)[0]
        if len(unvisited_customer_idxs) == 0:
            return
        print(unvisited_customer_idxs)
        exit()
        for cust_idx in unvisited_customer_idxs:
            vehicle_idxs, positions, insertion_costs = self.get_possible_insertions(cust_idx, solution)
            if len(vehicle_idxs) == 0:
                raise ValueError("No possible insertion found")
            is_reinsertion_feasible = False
            for vehicle_idx, pos_idx, insertion_cost in zip(vehicle_idxs, positions, insertion_costs):
                packing_result = self.try_insertion(cust_idx, vehicle_idx, pos_idx, solution)
                positions, rotations, is_packing_feasible = packing_result  
                if not is_packing_feasible:
                    continue
                is_reinsertion_feasible = True
                if len(solution.routes[vehicle_idx])==0:
                    solution.total_vehicle_fixed_cost += solution.vehicle_fixed_costs[vehicle_idx]
                    insertion_cost -= solution.vehicle_fixed_costs[vehicle_idx]
                solution.total_vehicle_variable_cost += insertion_cost    

                # now commit the route, because we can pack
                solution.cust_vhc_assignment_map[cust_idx] = vehicle_idx
                solution.filled_volumes[vehicle_idx] += solution.customer_demand_volumes[cust_idx]
                solution.filled_weight_caps[vehicle_idx] = solution.customer_demand_weights[cust_idx]
                solution.routes[vehicle_idx].insert(pos_idx, cust_idx)
                n = 0
                for ci in solution.routes[vehicle_idx]:
                    c_num_items = solution.problem.customers[ci].num_items
                    item_mask = solution.problem.customer_item_mask[ci, :]
                    solution.item_positions[item_mask] = positions[n:n+c_num_items]
                    solution.item_rotations[item_mask] = rotations[n:n+c_num_items]
                    n += c_num_items
                break
            if not is_reinsertion_feasible:
                raise ValueError(f"Reinsertion failed for {cust_idx}")
        return solution
        