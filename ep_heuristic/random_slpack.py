import random
from typing import Optional, Tuple

import numpy as np

from ep_heuristic.insertion import argsort_items, insert_items

def random_slpack(item_dims: np.ndarray,
                    item_volumes: np.ndarray,
                    item_priorities: np.ndarray,
                    container_dim: np.ndarray,
                    base_support_alpha: float,
                    max_trial:int,
                  ) ->Tuple[Optional[np.ndarray],Optional[np.ndarray],bool]:
    """
        item_priorities: the order of visitation, actually
    """
    num_items: int = len(item_dims)
    item_base_areas = np.prod(item_dims[:, :2], axis=1)
    sorted_idx = argsort_items(item_base_areas, item_volumes, item_priorities)
    rotation_trial_idx = np.tile(np.arange(2), [num_items, 1])
    for i in range(max_trial):
        if i>0:
            # swap some orderings as a local search operator
            j,k = random.randint(0, num_items-1), random.randint(0, num_items-1)
            if item_priorities[j] == item_priorities[k]:
                a = sorted_idx[j]
                sorted_idx[j] = sorted_idx[k]
                sorted_idx[k] = a
            
            j = random.randint(0, num_items-1)
            rotation_trial_idx[j, :] = rotation_trial_idx[j,[1,0]]
            
        positions, rotations, is_feasible = insert_items(item_dims[sorted_idx], container_dim, rotation_trial_idx, base_support_alpha)
        if not is_feasible:
            continue
        inverted_idx = np.argsort(sorted_idx)
        positions = positions[inverted_idx]
        rotations = rotations[inverted_idx]
        return positions, rotations, True
    return None, None, False