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