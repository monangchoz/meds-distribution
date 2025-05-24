from typing import List

import numpy as np

from problem.item import Item
from problem.node import Node

class Customer(Node):
    def __init__(self,
                 idx: int,
                 cust_id: str,
                 coord: np.ndarray,
                 items: List[Item]):
        super().__init__(idx, coord)
        self.cust_id: str = cust_id
        self.items: List[Item] = items
        self.num_items: int = len(items)
        self.need_refer_truck: bool = False
        for item in self.items:
            if item.is_reefer_required:
                self.need_refer_truck = True
                break
        
    def to_dict(self):
        d = {"idx": self.idx, "cust_id":self.cust_id, "coord":self.coord.tolist(), "items":[]}
        for item in self.items:
            d["items"] += [item.to_dict()]
        return d
    
    @classmethod
    def from_dict(cls, data: dict):
        items: List[Item] = []
        for item_dict in data["items"]:
            items.append(Item.from_dict(item_dict))
        return cls(data["idx"],
                   data["cust_id"],
                   np.asanyarray(data["coord"], dtype=float),
                   items)
        