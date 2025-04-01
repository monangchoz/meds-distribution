from typing import List, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

from problem.customer import Customer
from problem.node import Node
from problem.vehicle import Vehicle

class HVRP3L:
    def __init__(self,
                 depot_coord: np.ndarray,
                 customers: List[Customer],
                 vehicles: List[Vehicle],
                 distance_matrix: Optional[np.ndarray] = None):
        depot: Node= Node(0, depot_coord)
        self.customers: List[Customer] = customers
        self.nodes: List[Union[Node, Customer]] = [depot] + customers
        self.vehicles: List[Vehicle] = vehicles
        self.coords: np.ndarray = np.stack([node.coord for node in self.nodes], dtype=float)
        self.distance_matrix: np.ndarray
        if distance_matrix is None:
            self.distance_matrix = haversine_distances(self.coords, self.coords)
        else:
            self.distance_matrix = distance_matrix
        
        self.num_nodes: int = len(self.nodes)
        self.num_customers: int = len(self.customers)
        self.num_vehicles: int = len(self.vehicles)
        # okay from this on is information that are essential for solver
        self.total_demand_volumes: np.ndarray = np.zeros([self.num_customers,], dtype=float)
        self.total_demand_weights: np.ndarray = np.zeros([self.num_customers,], dtype=float)
        for ci, customer in enumerate(self.customers):
            total_volume = sum(item.volume for item in customer.items)
            self.total_demand_volumes[ci] = total_volume
            total_weight = sum(item.weight for item in customer.items)
            self.total_demand_weights[ci] = total_weight
            
        self.vehicle_volume_capacities: np.ndarray = np.asanyarray([vehicle.volume_capacity for vehicle in self.vehicles], dtype=float)
        self.vehicle_weight_capacities: np.ndarray = np.asanyarray([vehicle.weight_capacity for vehicle in self.vehicles], dtype=float)
        self.vehicle_container_dims: np.ndarray = np.zeros([self.num_vehicles, 3], dtype=float)
        for i, vehicle in enumerate(vehicles):
            self.vehicle_container_dims[i,:] = vehicle.container_dim
        self.vehicle_fixed_costs: np.ndarray = np.asanyarray([vehicle.fixed_cost for vehicle in vehicles])
        self.vehicle_variable_costs: np.ndarray = np.asanyarray([vehicle.variable_cost for vehicle in vehicles])
        
        # now for the items?
        # i dont know whether it is a good idea (enough merit) 
        # to flatten all of them here, or not.
        self.num_items: int = sum(customer.num_items for customer in customers)
        item_weights = []
        item_volumes = []
        item_dims = []
        item_fragility_flags = []
        for customer in customers:
            for item in customer.items:
                item_weights += [item.weight]
                item_volumes += [item.volume]
                item_dims += [item.dim]
                item_fragility_flags += [item.is_fragile]
        self.item_weights: np.ndarray = np.asanyarray(item_weights, dtype=float)
        self.item_volumes: np.ndarray = np.asanyarray(item_volumes, dtype=float)
        self.item_dims: np.ndarray = np.stack(item_dims)
        self.item_fragility_flags: np.ndarray = np.asanyarray(item_fragility_flags, dtype=bool) 

        self.customer_item_mask: np.ndarray = np.zeros([self.num_customers, self.num_items], dtype=bool)
        i = 0
        for ci, customer in enumerate(customers):
            for item in customer.items:
                self.customer_item_mask[ci, i] = True
                i += 1