import math

import numba as nb
import numpy as np


@np.vectorize(otypes=[bool])
def is_intersect_1d(start_point_a: float, 
                    length_a: float,
                    start_point_b: float, 
                    length_b: float)->bool:
    end_a, end_b = start_point_a+length_a, start_point_b+length_b
    return not (end_a <= start_point_b or end_b <= start_point_a)

@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]), cache=True)
def is_intersect_nd(start_point_a: np.ndarray,
                    dim_a: np.ndarray,
                    start_point_b: np.ndarray,
                    dim_b: np.ndarray)->np.bool:
    end_a, end_b = start_point_a+dim_a, start_point_b+dim_b
    is_intersect = np.logical_not(np.logical_or(end_a<=start_point_b, end_b<=start_point_a))
    is_intersect = np.all(is_intersect)
    return is_intersect

@nb.njit(nb.bool[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_intersect_nd_vectorized(start_point_a: np.ndarray,
                    dim_a: np.ndarray,
                    start_point_b: np.ndarray,
                    dim_b: np.ndarray)->np.bool:
    n_a, _ = start_point_a.shape
    n_b, _ = start_point_b.shape
    end_a, end_b = start_point_a+dim_a, start_point_b+dim_b
    is_intersect_all_dim = np.logical_not(np.logical_or(end_a[:, None, :]<=start_point_b[None, :, :], start_point_a[:, None, :]>=end_b[None, :, :]))
    is_intersect: np.ndarray = np.empty((n_a, n_b), dtype=np.bool_)
    for i in range(len(start_point_a)):
        for j in range(len(start_point_b)):
            is_intersect[i,j] = np.all(is_intersect_all_dim[i,j,:])
    # is_intersect = np.all(is_intersect, axis=-1)
    return is_intersect
    


@np.vectorize(otypes=[float])
def compute_intersection_1d(start_point_a: float, 
                            length_a: float,
                            start_point_b: float, 
                            length_b: float)->float:
    end_a, end_b = start_point_a+length_a, start_point_b+length_b
    # Find the intersection range
    intersection_start = max(start_point_a, start_point_b)
    intersection_end = min(end_a, end_b)
    
    # Compute intersection length without an explicit if-else
    return max(0.0, intersection_end - intersection_start)

@nb.njit(nb.float64(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]), cache=True)
def compute_intersection_nd(start_point_a: np.ndarray,
                            dim_a: np.ndarray,
                            start_point_b: np.ndarray,
                            dim_b: np.ndarray)->float:
    end_a, end_b = start_point_a+dim_a, start_point_b+dim_b
    intersection_start = np.maximum(start_point_a, start_point_b)
    intersection_end = np.minimum(end_a, end_b)
    intersection_length = intersection_end-intersection_start
    intersection_length = np.maximum(intersection_length, 0.)
    retv2 = np.prod(intersection_length)
    return retv2

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
def is_intersect_nd_any_v2(ep: np.ndarray,
                        dummy_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray)->bool:
    num_inserted_items, _ = inserted_item_dims.shape
    batch_size: int = 64
    num_batch: int = math.ceil(num_inserted_items/batch_size)
    for i in range(num_batch):
        ja:int = i*batch_size
        jb:int = min((i+1)*batch_size, num_inserted_items)
        if np.any(is_intersect_nd_vectorized(ep[None, :], dummy_dim[None, :], filled_positions[ja:jb], inserted_item_dims[ja:jb])):
            return True
    return False