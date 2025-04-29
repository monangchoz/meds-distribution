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
    
    batch_size = 4
    num_batches = math.ceil(max_trial/batch_size)
    # batched parallelization
    result_positions = np.empty((batch_size, num_items, 3), dtype=np.float64)
    result_rotations = np.empty((batch_size, num_items, 3), dtype=np.int64)
    result_feasibilities = np.zeros((batch_size,), dtype=np.bool_)
    actual_item_dims_tmp: np.ndarray = np.empty_like(result_positions) #this can change if rotated.

    sorted_idx_batches = np.empty((batch_size, num_items), dtype=np.int64)
    rotation_trial_idx_batches = np.empty((batch_size, num_items, 2), dtype=np.int64)
    for b in range(batch_size):
        sorted_idx_batches[b] = sorted_idx
        rotation_trial_idx_batches[b] = rotation_trial_idx

    for z in range(num_batches):
        for b in range(batch_size):
            if z*batch_size + b >= max_trial:
                break
            for t in range(5): # swapping 5 kali, maybe, i dont'know
                # swap some orderings as a local search operator
                j,k = np.random.randint(0, num_items-1, size=2)
                # j,k = random.randint(0, num_items-1), random.randint(0, num_items-1)
                if item_priorities[j] == item_priorities[k]:
                    a = sorted_idx_batches[b, j]
                    sorted_idx_batches[b,j] = sorted_idx_batches[b,k]
                    sorted_idx_batches[b,k] = a
                
                j = np.random.randint(0, num_items-1)
                
                rotation_trial_idx_batches[b, j] = 1 - rotation_trial_idx_batches[b, j]

            result_positions[b], result_rotations[b], result_feasibilities[b] = insert_items(item_dims[sorted_idx_batches[b]], 
                                                         container_dim,
                                                         possible_rotation_permutation_mats, 
                                                         rotation_trial_idx_batches[b],
                                                         result_positions[b],
                                                         result_rotations[b],
                                                         actual_item_dims_tmp[b],
                                                         base_support_alpha)
        for b in range(batch_size):
            if not result_feasibilities[b]:
                continue
            
            inverted_idx = np.argsort(sorted_idx_batches[b])
            positions = result_positions[b][inverted_idx]
            rotations = result_rotations[b][inverted_idx]
            return positions, rotations, True
    
    return result_positions[0], result_rotations[0], False    

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