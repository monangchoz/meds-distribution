import math
from typing import Tuple, Optional

import numba as nb
import numpy as np

from ep_heuristic.utils import is_intersect_nd_any, is_intersect_nd_any_v2, compute_intersection_nd, is_intersect_nd_vectorized
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

@nb.njit(nb.float64(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def compute_supported_base_area(item_dim: np.ndarray, 
                                insertion_position: np.ndarray, 
                                inserted_item_dims: np.ndarray, 
                                filled_positions: np.ndarray)->float:
    ret = 0.0
    for i in range(len(inserted_item_dims)):
        if insertion_position[2] != filled_positions[i][2]+inserted_item_dims[i][2]:
            continue
        ret += compute_intersection_nd(insertion_position[:2], item_dim[:2], filled_positions[i][:2], inserted_item_dims[i][:2])
    return ret

@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.float64), cache=True)
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
    insertion_results_in_intersection = is_intersect_nd_any_v2(insertion_position, item_dim, inserted_item_dims, filled_positions)
    if insertion_results_in_intersection:
        return False
        
    # base support
    
    if insertion_position[2]>0:
        supported_base_area = compute_supported_base_area(item_dim, insertion_position, inserted_item_dims, filled_positions)
        base_area = item_dim[0]*item_dim[1]
        if supported_base_area/base_area < base_support_alpha:
            return False
    # fragility
    
    # lifo? -> automatic from item ordering, later
        
    
    return True

# def is_insertion_feasible_vec(item_dim: np.ndarray,
#                           insertion_positions: np.ndarray,
#                           inserted_item_dims: np.ndarray,
#                           filled_positions: np.ndarray,
#                           container_dim: np.ndarray,
#                           base_support_alpha: float)-> bool:
#     is_out_of_container = np.any(item_dim[:, np.newaxis, :] + insertion_positions[] > container_dim)
#     if is_out_of_container:
#         return False
    
#     # check intersect any other items
#     insertion_results_in_intersection = is_intersect_nd_any(insertion_positions, item_dim, inserted_item_dims, filled_positions)
#     if insertion_results_in_intersection:
#         return False
        
#     # base support
    
#     if insertion_position[2]>0:
#         supported_base_area = compute_supported_base_area(item_dim, insertion_position, inserted_item_dims, filled_positions)
#         base_area = item_dim[0]*item_dim[1]
#         if supported_base_area/base_area < base_support_alpha:
#             return False
#     # fragility
    
#     # lifo? -> automatic from item ordering, later
        
    
#     return True

# @nb.njit(nb.int64(nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.float64), cache=True, parallel=True)
def find_ep(item_dim: np.ndarray,
            inserted_item_dims: np.ndarray,
            filled_positions: np.ndarray,
            ext_points: np.ndarray,
            container_dim: np.ndarray,
            base_support_alpha: float)-> int:
    # do it in batched
    batch_size:int = 5
    # total_size, _ = ext_points.shape
    # num_batch = math.ceil(total_size/batch_size)
    # if len(ext_points)>=batch_size:    
    #     iff_v2 = is_insertion_feasible_vec(item_dim.reshape((1,3)), ext_points[:batch_size], inserted_item_dims, filled_positions,  container_dim, base_support_alpha)
    #     is_insertion_feasible_flags = np.zeros((batch_size,), dtype=np.bool_)
    #     for i in range(batch_size):
    #         ep = ext_points[i]
    #         is_insertion_feasible_flags[i] = is_insertion_feasible(item_dim, ep, inserted_item_dims, filled_positions,  container_dim, base_support_alpha)
    
    for ei, ep in enumerate(ext_points):
        if is_insertion_feasible(item_dim, ep, inserted_item_dims, filled_positions,  container_dim, base_support_alpha):
            return ei
    # for i in range(num_batch):
    #     for ej in nb.prange(i*batch_size, min((i+1)*batch_size, total_size)):
    #         ep = ext_points[ej]
    #         j = ej - i*batch_size
    #         is_insertion_feasible_flags[j] = is_insertion_feasible(item_dim, ep, inserted_item_dims, filled_positions,  container_dim, base_support_alpha)
    #     for j in range(batch_size):
    #         ej = i*batch_size + j
    #         if ej >= total_size:
    #             break
    #         if is_insertion_feasible_flags[j] == True:
    #             return ej
    return -1


@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_ep_floating(ep: np.ndarray,
                   dummy_dim: np.ndarray,
                    inserted_item_dims: np.ndarray,
                    filled_positions: np.ndarray)->bool:
    if ep[2]==0:
        return False
    if is_intersect_nd_any_v2(ep[:2], dummy_dim[:2], inserted_item_dims[:, :2], filled_positions[:, :2]):
        return False
    return True


@nb.njit(nb.bool[:](nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def get_ep_feasibility_mask(dummy_dim: np.ndarray,
                            ext_points: np.ndarray,
                            inserted_item_dims: np.ndarray,
                            filled_positions: np.ndarray)->np.ndarray:
    num_ep,_ = ext_points.shape
    ep_feasibility_mask: np.ndarray = np.empty((num_ep,), dtype=np.bool_)
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

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def filter_infeasible_extreme_points(ext_points: np.ndarray,
                                     dummy_dim: np.ndarray,
                                     inserted_item_dims: np.ndarray,
                                     filled_positions: np.ndarray)->np.ndarray:    
    ep_feasibility_mask = get_ep_feasibility_mask(dummy_dim,
                                                    ext_points,
                                                    inserted_item_dims,
                                                    filled_positions)
    ext_points = ext_points[ep_feasibility_mask]
    return ext_points
    
@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:]), cache=True)
def merge_and_lexsort_and_unique(ext_points: np.ndarray, new_ext_points: np.ndarray)->np.ndarray:
    # this assumes the two arrays are sorted
    n_a, _ = ext_points.shape
    n_b, _ = new_ext_points.shape
    n_c = 0
    final_ext_points = np.empty((n_a+n_b,3), dtype=np.float64)
    p_a = 0
    p_b = 0
    while p_a < n_a and p_b < n_b:
        ep_a, ep_b = ext_points[p_a], new_ext_points[p_b]
        
        if ep_a[2] < ep_b[2] or (ep_a[2] == ep_b[2] and ep_a[0] < ep_b[0]) or (ep_a[2] == ep_b[2] and ep_a[0] == ep_b[0] and ep_a[1] < ep_b[1]):
            new_ep = ep_a
            p_a += 1
        else:
            new_ep = ep_b
            p_b += 1
        
        # Only add if the current point is not a duplicate
        if n_c == 0 or not np.all(final_ext_points[n_c - 1] == new_ep):
            final_ext_points[n_c] = new_ep
            n_c += 1
    
    # Add remaining points from ext_points, if any
    while p_a < n_a:
        new_ep = ext_points[p_a]
        if n_c == 0 or not np.all(final_ext_points[n_c - 1] == new_ep):
            final_ext_points[n_c] = new_ep
            n_c += 1
        p_a += 1
    
    # Add remaining points from new_ext_points, if any
    while p_b < n_b:
        new_ep = new_ext_points[p_b]
        if n_c == 0 or not np.all(final_ext_points[n_c - 1] == new_ep):
            final_ext_points[n_c] = new_ep
            n_c += 1
        p_b += 1
    
    # Ensure we return all the merged and unique points
    return final_ext_points[:n_c]

@nb.njit(nb.float64[:,:](nb.float64[:],nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:]), cache=True)
def update_extreme_points_from_new_item(item_dim: np.ndarray,
                                        chosen_ep_idx: int,
                                        inserted_item_dims: np.ndarray,
                                        filled_positions: np.ndarray,
                                        ext_points: np.ndarray,
                                        dummy_dim: np.ndarray,
                                        # construction_mode: str="wall-building"
                                        )->np.ndarray:
    # final_ext_points = np.empty([len(ext_points)+6, 3],  dtype=float)
    new_item_position = ext_points[chosen_ep_idx]
    new_ext_points = new_item_position[None, :] + np.diag(item_dim)
    # new_ext_points[[0,1],:] = new_ext_points[[1,0],:] # this is wall-building lexsort
    # swap the first and second row
    temp = new_ext_points[0].copy()
    new_ext_points[0] = new_ext_points[1]
    new_ext_points[1] = temp
    # TODO add projection
    new_ext_points = filter_infeasible_extreme_points(new_ext_points, dummy_dim, inserted_item_dims, filled_positions)
    ep_feasibility_mask:np.ndarray = np.logical_not(is_intersect_nd_vectorized(ext_points, dummy_dim[None, :], filled_positions[-1:], inserted_item_dims[-1:])).ravel()
    ext_points = ext_points[ep_feasibility_mask]
    final_ext_points = merge_and_lexsort_and_unique(ext_points, new_ext_points)
    return final_ext_points
    
@profile
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
                ext_points = update_extreme_points_from_new_item(item_dim, ei, inserted_item_dims, filled_positions, ext_points, DUMMY_DIM)
                break
        if found_feasible_ep:
            continue
        # infeasible
        return None, None, False
    
    return positions, rotations, True
        