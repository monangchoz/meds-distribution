import copy
import math
from typing import List, Optional

import numpy as np
from ep_heuristic.utils import is_packing_feasible
from problem.hvrp3l import HVRP3L

NO_VEHICLE:int = 99999
DEPOT:int = 0
DEPOT_DONT_CARE: int = 99998


class Solution:
    def __init__(self,
                 hvrp3l_instance: HVRP3L,
                 filled_volumes: Optional[np.ndarray] = None,
                 filled_weight_caps: Optional[np.ndarray] = None,
                 item_positions: Optional[np.ndarray] = None,
                 item_rotations: Optional[np.ndarray] = None,
                 routes: Optional[List[List[int]]] = None,
                 node_vhc_assignment_map: Optional[np.ndarray] = None,
                 total_vehicle_variable_cost: float = 0,
                 total_vehicle_fixed_cost: float = 0,
                 ):
        self.problem: HVRP3L = hvrp3l_instance
        self.num_nodes = hvrp3l_instance.num_nodes
        self.num_customers = hvrp3l_instance.num_customers
        self.num_vehicles = hvrp3l_instance.num_vehicles
        self.num_items = hvrp3l_instance.num_items
        
        self.node_num_items: np.ndarray = hvrp3l_instance.node_num_items
        self.node_demand_volumes: np.ndarray = hvrp3l_instance.total_demand_volumes
        self.node_demand_weights: np.ndarray = hvrp3l_instance.total_demand_weights
        self.node_reefer_flags: np.ndarray = hvrp3l_instance.node_reefer_flags
        self.vehicle_volume_capacities: np.ndarray = hvrp3l_instance.vehicle_volume_capacities
        self.vehicle_weight_capacities: np.ndarray = hvrp3l_instance.vehicle_weight_capacities
        self.vehicle_container_dims: np.ndarray = hvrp3l_instance.vehicle_container_dims
        self.vehicle_reefer_flags: np.ndarray = hvrp3l_instance.vehicle_reefer_flags
    
        self.vehicle_fixed_costs: np.ndarray = hvrp3l_instance.vehicle_fixed_costs
        self.vehicle_variable_costs: np.ndarray = hvrp3l_instance.vehicle_variable_costs
        self.item_weights: np.ndarray = hvrp3l_instance.item_weights
        self.item_volumes: np.ndarray = hvrp3l_instance.item_volumes
        self.item_dims: np.ndarray = hvrp3l_instance.item_dims
        self.item_fragility_flags: np.ndarray = hvrp3l_instance.item_fragility_flags
        self.node_item_mask: np.ndarray = hvrp3l_instance.node_item_mask
        
        # decision variables
        # all must be provided if copying, otherwise, none for all
        if filled_volumes is not None:
            self.filled_volumes = filled_volumes.copy()
            self.filled_weight_caps = filled_weight_caps.copy()
            self.item_positions = item_positions.copy()
            self.item_rotations = item_rotations.copy()
            self.routes = copy.deepcopy(routes)
            self.node_vhc_assignment_map = node_vhc_assignment_map.copy()
            self.total_vehicle_variable_cost = total_vehicle_variable_cost
            self.total_vehicle_fixed_cost = total_vehicle_fixed_cost
        else:
            self.filled_volumes: np.ndarray = np.zeros([self.num_vehicles,], dtype=float)
            self.filled_weight_caps: np.ndarray = np.zeros([self.num_vehicles,], dtype=float)
            self.item_positions: np.ndarray = np.zeros_like(self.item_dims)
            self.item_rotations: np.ndarray = np.zeros_like(self.item_dims, dtype=int)
            self.routes: List[List[int]] = [[] for _ in range(self.num_vehicles)]
            self.node_vhc_assignment_map: np.ndarray = np.full([self.num_nodes,], NO_VEHICLE, dtype=int)
            self.node_vhc_assignment_map[DEPOT] = DEPOT_DONT_CARE
            self.total_vehicle_variable_cost:float = 0
            self.total_vehicle_fixed_cost:float = 0

    def copy(self):
        return self.__class__(self.problem,
                              self.filled_volumes,
                              self.filled_weight_caps,
                              self.item_positions,
                              self.item_rotations,
                              self.routes,
                              self.node_vhc_assignment_map,
                              self.total_vehicle_variable_cost,
                              self.total_vehicle_fixed_cost)


    @property
    def total_cost(self):
        ret = self.total_vehicle_fixed_cost + self.total_vehicle_variable_cost
        ret = math.trunc(ret*1000)/1000
        return ret
    
    @property
    def total_distance(self):
        ret = 0
        for vi in range(self.num_vehicles):
            ret += self.problem.compute_route_total_distance(self.routes[vi])
        return ret
    
    @property
    def is_feasible(self):
        total_fixed_cost = 0
        total_vehicle_variable_cost = 0
        
        is_visited = np.zeros((self.num_nodes, ), dtype=bool)
        is_visited[0] = True
        for vi, route in enumerate(self.routes):
            for cust_idx in route:
                assert not is_visited[cust_idx]
                is_visited[cust_idx] = True
                assert not self.node_reefer_flags[cust_idx] or self.vehicle_reefer_flags[vi]

        assert np.sum(is_visited) == self.num_nodes
        filled_volumes = np.zeros((self.num_vehicles,), dtype=float)
        filled_weights = np.zeros((self.num_vehicles,), dtype=float)
        for vi, route in enumerate(self.routes):
            if len(route)==0:
                continue
            total_fixed_cost += self.vehicle_fixed_costs[vi]
            distance = 0
            prev_node = 0
            for cust_idx in route:
                distance += self.problem.distance_matrix[prev_node, cust_idx]
                prev_node = cust_idx
            distance += self.problem.distance_matrix[prev_node, 0]
            total_vehicle_variable_cost += distance*self.vehicle_variable_costs[vi]
            
            for cust_idx in route:
                filled_volumes[vi] += self.node_demand_volumes[cust_idx]
                filled_weights[vi] += self.node_demand_weights[cust_idx]
            assert filled_volumes[vi] <= self.vehicle_volume_capacities[vi]
            assert filled_weights[vi] <= self.vehicle_weight_capacities[vi]
            
            total_num_items = sum(self.node_num_items[cust_idx] for cust_idx in route)
            item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
            positions: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
            rotations: np.ndarray = np.zeros([total_num_items, 3], dtype=int)
            n = 0
            for i, cust_idx in enumerate(route):
                c_num_items = self.node_num_items[cust_idx]
                item_mask = self.node_item_mask[cust_idx, :]
                item_dims[n:n+c_num_items] = self.item_dims[item_mask]
                positions[n:n+c_num_items] = self.item_positions[item_mask]
                rotations[n:n+c_num_items] = self.item_rotations[item_mask]
                n += c_num_items
            assert is_packing_feasible(self.vehicle_container_dims[vi], item_dims, rotations, positions)        
        print(self.total_vehicle_fixed_cost, total_fixed_cost)
        assert math.isclose(self.total_vehicle_fixed_cost, total_fixed_cost)
        assert math.isclose(self.total_vehicle_variable_cost, total_vehicle_variable_cost)
        return True