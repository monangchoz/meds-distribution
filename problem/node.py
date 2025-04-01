import numpy as np

class Node:
    def __init__(self, 
                 idx: int,
                 coord: np.ndarray):
        self.idx: int = idx
        self.coord: np.ndarray = coord