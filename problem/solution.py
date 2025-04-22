from typing import List

import numpy as np
from problem.hvrp3l import HVRP3L

NO_VEHICLE:int = 99999
DEPOT:int = 0
DEPOT_DONT_CARE: int = 99998


class Solution:
    def __init__(self,
                 hvrp3l_instance: HVRP3L):
        self.problem: HVRP3L = hvrp3l_instance
        self.num_nodes = hvrp3l_instance.num_nodes
        self.num_customers = hvrp3l_instance.num_customers
        self.num_vehicles = hvrp3l_instance.num_vehicles
        self.num_items = hvrp3l_instance.num_items
        self.routes: List[List[int]] = [[] for _ in range(self.num_vehicles)]
        self.node_vhc_assignment_map: np.ndarray = np.full([self.num_nodes,], NO_VEHICLE, dtype=int)
        self.node_vhc_assignment_map[DEPOT] = DEPOT_DONT_CARE
        
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
        
        self.filled_volumes: np.ndarray = np.zeros([self.num_vehicles,], dtype=float)
        self.filled_weight_caps: np.ndarray = np.zeros([self.num_vehicles,], dtype=float)
        self.item_positions: np.ndarray = np.zeros_like(self.item_dims)
        self.item_rotations: np.ndarray = np.zeros_like(self.item_dims, dtype=int)

        self.total_vehicle_variable_cost:float = 0
        self.total_vehicle_fixed_cost:float = 0

    @property
    def total_cost(self):
        return self.total_vehicle_fixed_cost + self.total_vehicle_variable_cost