import random
from typing import Tuple, Optional

import numba as nb
import numpy as np

from ep_heuristic.utils import is_intersect_nd, compute_intersection_nd
from problem.item import POSSIBLE_ROTATION_PERMUTATION_MATS


def argsort_items(item_base_areas: np.ndarray,
                 item_volumes: np.ndarray,
                 item_priorities: np.ndarray)->np.ndarray:
    """10.1109/MCI.2014.2350933 page 26, some rules are still omitted

    Args:
        item_base_areas (np.ndarray): _description_
        item_volumes (np.ndarray): _description_
        item_priorities (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    sorted_idx = np.lexsort((item_volumes, item_base_areas, item_priorities))
    return sorted_idx

def is_insertion_feasible(item_dim: np.ndarray,
                          insertion_position: np.ndarray,
                          inserted_item_dims: np.ndarray,
                          filled_positions: np.ndarray,
                          container_dim: np.ndarray,
                          base_support_alpha: float)-> bool:
    is_out_of_container = np.any(insertion_position + item_dim > container_dim)
    if is_out_of_container:
        return False
    
    # check intersect any other items
    num_inserted_items = len(inserted_item_dims)
    for i in range(num_inserted_items):
        if is_intersect_nd(insertion_position, item_dim, filled_positions[i], inserted_item_dims[i]):
            return False
        
    # base support
    if insertion_position[2]>0:
        supported_base_area = 0
        base_area = item_dim[0]*item_dim[1]
        for i in range(num_inserted_items):
            if insertion_position[2] != filled_positions[i][2]+inserted_item_dims[i][2]:
                continue
            supported_base_area += compute_intersection_nd(insertion_position[:2], item_dim[:2], filled_positions[i][:2], inserted_item_dims[i][:2])
        if supported_base_area/base_area < base_support_alpha:
            return False

    # fragility
    
    # lifo? -> automatic from item ordering, later
        
    
    return True
    
def find_ep(item_dim: np.ndarray,
            inserted_item_dims: np.ndarray,
            filled_positions: np.ndarray,
            ext_points: np.ndarray,
            container_dim: np.ndarray,
            base_support_alpha: float)-> int:
    for ei, ep in enumerate(ext_points):
        if is_insertion_feasible(item_dim, ep, inserted_item_dims, filled_positions,  container_dim, base_support_alpha):
            return ei
    return -1


@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_intersect_nd_any(ep: np.ndarray,
                        dummy_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray)->bool:
    for i, inserted_item_dim in enumerate(inserted_item_dims):
        filled_position = filled_positions[i]
        # check if placing a very very small item here intersect with other inserted items
        if is_intersect_nd(ep, dummy_dim, filled_position, inserted_item_dim):
            return True
    return False

@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_ep_floating(ep: np.ndarray,
                   dummy_dim: np.ndarray,
                    inserted_item_dims: np.ndarray,
                    filled_positions: np.ndarray)->bool:
    if ep[2]==0:
        return False
    if is_intersect_nd_any(ep[:2], dummy_dim[:2], inserted_item_dims[:, :2], filled_positions[:, :2]):
        return False
    return True

# @nb.njit(nb.bool(nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def are_eps_floating(ext_points: np.ndarray,
                   dummy_dim: np.ndarray,
                    inserted_item_dims: np.ndarray,
                    filled_positions: np.ndarray)->np.bool:
    ep_floating_mask: np.ndarray = ext_points[:, 2] > 0
    ep_floating_mask[ep_floating_mask] = np.logical_not(are_intersect_nd_any(ext_points[ep_floating_mask, :2], dummy_dim[:2, inserted_item_dims[:2], filled_positions[:2]]))
    return True

@nb.njit(nb.bool[:](nb.bool[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def get_ep_feasibility_mask(ep_feasibility_mask: np.ndarray,
                                     dummy_dim: np.ndarray,
                                     ext_points: np.ndarray,
                                     inserted_item_dims: np.ndarray,
                                     filled_positions: np.ndarray)->np.ndarray:
    num_ep = len(ext_points)
    for ei in range(num_ep):
        if is_intersect_nd_any(ext_points[ei], dummy_dim, inserted_item_dims, filled_positions):
            ep_feasibility_mask[ei] = False
            continue
        
        if is_ep_floating(ext_points[ei], dummy_dim, inserted_item_dims, filled_positions):
            ep_feasibility_mask[ei] = False
            continue
        ep_feasibility_mask[ei] = True
    return ep_feasibility_mask

# @nb.njit(nb.bool[:](nb.bool[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def get_ep_feasibility_mask_v2(ep_feasibility_mask: np.ndarray,
                                     dummy_dim: np.ndarray,
                                     ext_points: np.ndarray,
                                     inserted_item_dims: np.ndarray,
                                     filled_positions: np.ndarray)->np.ndarray:
    num_ep = len(ext_points)
    for ei in range(num_ep):
        if is_intersect_nd_any(ext_points[ei], dummy_dim, inserted_item_dims, filled_positions):
            ep_feasibility_mask[ei] = False
            continue
        
        if is_ep_floating(ext_points[ei], dummy_dim, inserted_item_dims, filled_positions):
            ep_feasibility_mask[ei] = False
            continue
        ep_feasibility_mask[ei] = True
    return ep_feasibility_mask

DUMMY_DIM = np.asanyarray([0.0001, 0.0001, 0.0001], dtype=float)
def filter_infeasible_extreme_points(ext_points: np.ndarray,
                                     inserted_item_dims: np.ndarray,
                                     filled_positions: np.ndarray)->np.ndarray:    
    ep_feasibility_mask:np.ndarray = np.ones([len(ext_points),], dtype=bool)
    ep_feasibility_mask = get_ep_feasibility_mask(ep_feasibility_mask, 
                                                     DUMMY_DIM,
                                                     ext_points,
                                                     inserted_item_dims,
                                                     filled_positions)
    # remove infeasible
    ext_points = ext_points[ep_feasibility_mask]
    return ext_points
    
def update_extreme_points_from_new_item(item_dim: np.ndarray,
                                        chosen_ep_idx: int,
                                        inserted_item_dims: np.ndarray,
                                        filled_positions: np.ndarray,
                                        ext_points: np.ndarray,
                                        construction_mode: str="wall-building")->np.ndarray:
    new_item_position = ext_points[chosen_ep_idx]
    new_ext_points = new_item_position[None, :] + np.diag(item_dim)
    # TODO add projection
    ext_points = np.concatenate([ext_points, new_ext_points], axis=0)
    ext_points = filter_infeasible_extreme_points(ext_points, inserted_item_dims, filled_positions)
    if construction_mode == "wall-building":
        sorted_ep_idx = np.lexsort([ext_points[:, 1], ext_points[:, 0], ext_points[:, 2]])
    return ext_points[sorted_ep_idx]
    
def insert_items(item_dims: np.ndarray,
                 container_dim: np.ndarray,
                 rotation_trial_idx: Optional[np.ndarray] = None,
                 base_support_alpha: float = 0.6
    )->Tuple[Optional[np.ndarray],Optional[np.ndarray],bool]:
    num_items: int = len(item_dims)
    positions: np.ndarray = np.zeros([num_items, 3], dtype=float)
    rotations: np.ndarray = np.zeros([num_items, 3], dtype=int)
    actual_item_dims: np.ndarray = np.zeros_like(item_dims) #this can change if rotated.
    if rotation_trial_idx is None:
        rotation_trial_idx = np.tile(np.arange(2), [num_items, 1])
    ext_points: np.ndarray = np.zeros([1,3], dtype=float)
    for i in range(num_items):
        found_feasible_ep: bool = False
        filled_positions = positions[:i]
        inserted_item_dims = actual_item_dims[:i]
        for ri in rotation_trial_idx[i]:
            rotation = POSSIBLE_ROTATION_PERMUTATION_MATS[ri]
            item_dim = item_dims[i][rotation]
            ei = find_ep(item_dim, inserted_item_dims, filled_positions, ext_points, container_dim, base_support_alpha)
            if ei != -1:
                found_feasible_ep = True
                rotations[i] = rotation
                actual_item_dims[i] = item_dim
                positions[i] = ext_points[ei]
                filled_positions = positions[:i+1]
                inserted_item_dims = actual_item_dims[:i+1]
                ext_points = update_extreme_points_from_new_item(item_dim, ei, inserted_item_dims, filled_positions, ext_points)
                break
        if found_feasible_ep:
            continue
        # infeasible
        return None, None, False
    
    return positions, rotations, True
        