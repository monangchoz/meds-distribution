from typing import List, Tuple

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from problem.solution import Solution

def try_packing_custs_in_route(solution: Solution, 
                               vi: int, 
                               route:List[int])->Tuple[np.ndarray, np.ndarray, bool]:
    problem = solution.problem
    total_num_items = np.sum(solution.node_num_items[route])
            
    # this all actually can be pre-allocated in the problem interface
    # and used freely, to remove allocation time
    item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
    item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=float)
    n = 0
    for i, cust_idx in enumerate(route):
        c_num_items = solution.node_num_items[cust_idx]
        item_mask = problem.node_item_mask[cust_idx, :]
        item_dims[n:n+c_num_items] = problem.item_dims[item_mask]
        item_volumes[n:n+c_num_items] = problem.item_volumes[item_mask]
        item_weights[n:n+c_num_items] = problem.item_weights[item_mask]
        item_priorities[n:n+c_num_items] = i
        n += c_num_items
            
        # let's try packing
    container_dim = problem.vehicle_container_dims[vi]
    packing_result = random_slpack(item_dims,
                                    item_volumes,
                                    item_priorities,
                                    container_dim,
                                    0.8,
                                    5)
    return packing_result
