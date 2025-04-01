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
        
    def to_dict(self):
        vehicle_dict = {"idx": self.idx,
                        "vehicle_type": self.vehicle_type,
                        "weight_capacity": self.weight_capacity,
                        "container_dim": self.container_dim.tolist(),
                        "is_reefer": self.is_reefer,
                        "fixed_cost": self.fixed_cost,
                        "variable_cost": self.variable_cost,
                        }
        return vehicle_dict
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["idx"],
                data["vehicle_type"],
                data["weight_capacity"],
                np.asanyarray(data["container_dim"], dtype=float),
                data["is_reefer"],
                data["fixed_cost"],
                data["variable_cost"],
        )