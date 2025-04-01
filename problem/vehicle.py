import numpy as np


class Vehicle:
    def __init__(self,
                 idx: int,
                 vehicle_type: str,
                 weight_capacity: float,
                 container_dim: np.ndarray,
                 is_reefer: bool,
                 fixed_cost: float,
                 variable_cost: float):
        self.idx: int = idx
        self.vehicle_type: str = vehicle_type
        self.weight_capacity: float = weight_capacity
        self.container_dim: np.ndarray = container_dim
        self.volume_capacity = np.prod(container_dim)
        self.is_reefer: bool = is_reefer
        self.fixed_cost: float = fixed_cost
        self.variable_cost: float = variable_cost