from typing import List

import numpy as np

POSSIBLE_ROTATION_PERMUTATION_MATS: List[np.ndarray] =\
    [np.asanyarray([0,1,2], dtype=int), np.asanyarray([1,0,2], dtype=int)]

class Item:
    """
        an item or a medicine
    """
    def __init__(self,
                 idx: int,
                 item_type: str,
                 dim: np.ndarray,
                 weight: float,
                 is_fragile: bool,
                 is_reefer_required: bool):
        """

        Args:
            idx (int): unique index just to differentiate between
            this item and other items if they are in a list, i.e., 
            a customer has a list of items.
            item_type (str): item_id from the item_type list
            dim (np.ndarray): np(3,) float
            weight (float): weight
            is_fragile (bool): _description_
            is_reefer_required (bool): reefer required for cold item/med
        """
        self.idx: int = idx
        self.item_type: str = item_type
        self.dim: np.ndarray = dim
        self.weight: float = weight
        self.volume: float = np.prod(dim)
        self.is_fragile: bool = is_fragile
        self.is_reefer_required: bool = is_reefer_required
        
    def to_dict(self):
        item_dict = {"idx": self.idx,
            "product_code":self.item_type,
            "dim": self.dim.tolist(),
            "weight": self.weight,
            "volume": self.volume,
            "is_reefer_required": self.is_reefer_required,
            "is_fragile": self.is_fragile}
        return item_dict
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["idx"],
                   data["product_code"],
                   np.asanyarray(data["dim"], dtype=float),
                   data["weight"],
                   data["is_fragile"],
                   data["is_reefer_required"])
        
        
        
    
    
    