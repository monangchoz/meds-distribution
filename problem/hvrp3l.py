import json 
import os
import pathlib
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
        self.depot: Node= Node(0, depot_coord)
        self.customers: List[Customer] = customers
        self.nodes: List[Union[Node, Customer]] = [self.depot] + customers
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
                
    def to_json(self, cabang:str): 
        instance_dir = pathlib.Path()/"instances"
        instance_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{cabang}_nc_{self.num_customers}_ni__{self.num_items}_nv_{self.num_vehicles}"
        instance_filepath = ""
        for i in range(100000):
            instance_filepath = instance_dir/f"{filename}_{i}.json"
            if not os.path.isfile(instance_filepath.absolute()):
                break
        instance_dict = {}
        instance_dict["depot_coord"] = self.depot.coord.tolist()
        instance_dict["customers"] = {}
        for customer in self.customers:
            instance_dict["customers"][customer.cust_id] = customer.to_dict()
        instance_dict["vehicles"] = []
        for vehicle in self.vehicles:
            instance_dict["vehicles"] += [vehicle.to_diclt()]
    
        with open(instance_filepath.absolute(), "w") as f:
            json.dump(instance_dict, f)
            
    @classmethod
    def read_from_json(cls, json_filepath: pathlib.Path):
        with open(json_filepath.absolute(), "r") as json_data:
            d = json.load(json_data)
        depot_coord = np.asanyarray(d["depot_coord"], dtype=float)
        customers: List[Customer] = []
        for customer_dict in d["customers"]:
            new_customer: Customer = Customer.from_dict(customer_dict)
            customers.append(new_customer)
        customers = sorted(customers, key=lambda cust: cust.idx)
        
        vehicles: List[Vehicle] = []
        for vehicle_dict in d["vehicles"]:
            new_vehicle: Vehicle = Vehicle.from_dict(vehicle_dict)
            vehicles.append(new_vehicle)
        vehicles = sorted(vehicles, key=lambda vehicle: vehicle.idx)
        
        problem = cls(depot_coord, customers, vehicles)
        return problem
    