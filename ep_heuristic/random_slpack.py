import math
import random
from typing import Optional, Tuple

import numba as nb
import numpy as np
from ep_heuristic.insertion import argsort_items, insert_items
from line_profiler import profile
from problem.item import POSSIBLE_ROTATION_PERMUTATION_MATS


@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.int64[:,:], nb.bool))
         (nb.float64[:,:], nb.int64[:], nb.int64[:], nb.int64[:,:], nb.float64[:], nb.int64[:,:], nb.float64, nb.int64))
def try_slpack(item_dims: np.ndarray,
                item_priorities: np.ndarray,
                sorted_idx: np.ndarray,
                rotation_trial_idx: np.ndarray,
                container_dim: np.ndarray,
                possible_rotation_permutation_mats: np.ndarray,
                base_support_alpha: float,
                max_trial:int,
                  ) ->Tuple[np.ndarray,np.ndarray,bool]:
    num_items = len(item_dims)
    
    positions = np.empty((num_items, 3), dtype=np.float64)
    rotations = np.empty((num_items, 3), dtype=np.int64)
    actual_item_dims: np.ndarray = np.empty_like(positions) #this can change if rotated.

    for i in range(max_trial):
        if i>0:
            # swap some orderings as a local search operator
            j,k = random.randint(0, num_items-1), random.randint(0, num_items-1)
            if item_priorities[j] == item_priorities[k]:
                a = sorted_idx[j]
                sorted_idx[j] = sorted_idx[k]
                sorted_idx[k] = a
            
            j = random.randint(0, num_items-1)
            rotation_trial_idx[j] = 1 - rotation_trial_idx[j]
            
        positions, rotations, is_feasible = insert_items(item_dims[sorted_idx], 
                                                         container_dim, 
                                                         possible_rotation_permutation_mats, 
                                                         rotation_trial_idx,
                                                         positions,
                                                         rotations,
                                                         actual_item_dims, 
                                                         base_support_alpha)
        if not is_feasible:
            continue
        inverted_idx = np.argsort(sorted_idx)
        positions = positions[inverted_idx]
        rotations = rotations[inverted_idx]
        return positions, rotations, True
            
    return positions, rotations, False    


def random_slpack(item_dims: np.ndarray,
                    item_volumes: np.ndarray,
                    item_priorities: np.ndarray,
                    container_dim: np.ndarray,
                    base_support_alpha: float,
                    max_trial:int,
                  ) ->Tuple[np.ndarray,np.ndarray,bool]:
    """
        item_priorities: the order of visitation, actually
    """
    num_items = len(item_dims)
    item_base_areas = item_dims[:, 0]*item_dims[:, 1]
    sorted_idx = argsort_items(item_base_areas, item_volumes, item_priorities)
    rotation_trial_idx = np.tile(np.arange(2), [num_items, 1])
    positions, rotations, is_feasible = try_slpack(item_dims, item_priorities, sorted_idx, rotation_trial_idx, container_dim, POSSIBLE_ROTATION_PERMUTATION_MATS, base_support_alpha, max_trial)
    return positions, rotations, is_feasible