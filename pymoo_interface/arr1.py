import numpy as np
from typing import List, Tuple


from problem.solution import Solution, NO_VEHICLE

class RepairMechanism:
    def __init__(self, 
                 num_customers: int, 
                 num_vehicles: int):
        num_possible_positions = num_customers + num_vehicles*2
        self.vehicle_idxs = np.empty((num_possible_positions,), dtype=int)
        self.positions = np.empty((num_possible_positions,), dtype=int)
        self.insertion_costs = np.empty((num_possible_positions,), dtype=float)
        self.tmp_routes = np.empty((num_vehicles, num_customers+1), dtype=int)
        

    def repair(self, solution: Solution):
        raise NotImplementedError



# we can wrap it, closure, so routes can be numpy ndarray always
def get_sorted_possible_insertion_positions(cust_idx:int, 
                                            cust_demand_volume: float,
                                            cust_demand_weight: float,
                                            customer_need_reefer: bool, 
                                            vehicle_filled_volumes: np.ndarray,
                                            vehicle_filled_weights: np.ndarray,
                                            vehicle_volumes: np.ndarray,
                                            vehicle_weight_capacities: np.ndarray,
                                            vehicle_reefer_flags: np.ndarray,
                                            routes: List[List[int]],
                                            distance_matrix: np.ndarray,
                                            vehicle_fixed_costs: np.ndarray,
                                            vehicle_variable_costs: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vehicle_idxs = []
    positions = []
    insertion_costs = []
    for vehicle_idx, route in enumerate(routes):
        vehicle_has_enough_volume = cust_demand_volume + vehicle_filled_volumes[vehicle_idx] <= vehicle_volumes[vehicle_idx]
        vehicle_has_enough_capacity = cust_demand_weight + vehicle_filled_weights[vehicle_idx] <= vehicle_weight_capacities[vehicle_idx]
        if not (vehicle_has_enough_volume and vehicle_has_enough_capacity):
            continue
        if customer_need_reefer and not vehicle_reefer_flags[vehicle_idx]:
            continue
        route_len = len(route)
        for pos_idx in range(route_len+1):
            if pos_idx == 0:
                prev_node = 0
            else:
                prev_node = route[pos_idx-1]
            
            if pos_idx == route_len:
                next_node = 0
            else:
                next_node = route_len[pos_idx]
            

            d_distance = distance_matrix[prev_node, cust_idx] + distance_matrix[cust_idx, next_node] - distance_matrix[prev_node, next_node]
            insertion_cost = d_distance*vehicle_variable_costs[vehicle_idx]
            if route_len == 0:
                insertion_cost += vehicle_fixed_costs[vehicle_idx]

            vehicle_idxs.append(vehicle_idx)
            positions.append(pos_idx)
            insertion_costs.append(insertion_cost)
    
    vehicle_idxs = np.asanyarray(vehicle_idxs) 
    positions = np.asanyarray(positions) 
    insertion_costs = np.asanyarray(insertion_costs)
    sorted_idx = np.argsort(insertion_costs)
    vehicle_idxs = vehicle_idxs[sorted_idx]
    positions = positions[sorted_idx]
    insertion_costs = insertion_costs[sorted_idx]
    return vehicle_idxs, positions, insertion_costs



class ARR1(RepairMechanism):
    def repair(self, solution: Solution):
        # print(solution.cust_vhc_assignment_map)
        unvisited_customer_idxs: np.ndarray = np.nonzero(solution.cust_vhc_assignment_map==NO_VEHICLE)[0]
        if len(unvisited_customer_idxs) == 0:
            return
        for cust_idx in unvisited_customer_idxs:
            
            
            # insert into first feasible? whats the best way here? sort insert positions based on insertion cost
            # and then check if feasible.. right?   
        print("HELLO", unvisited_customer_idxs)
        exit()
    # for ci in unvisited_customer_idxs:
    #     # try to visit to a vehicle, if not possible, then 
    #     # return infeasible?
        