import math

import numba as nb
import numpy as np


def is_packing_feasible(container_dim: np.ndarray,
                        item_dims: np.ndarray, 
                        rotations: np.ndarray,
                        positions: np.ndarray)->bool:
    item_actual_dims = np.take_along_axis(item_dims, rotations, axis=1)
    num_items, _ = item_dims.shape
    for i in range(num_items):
        if is_outside_container(positions[i], item_actual_dims[i], container_dim):
            return False
        if is_intersect_nd_any_v2(positions[i], item_actual_dims[i], item_actual_dims[:i], positions[:i]):
            return False
    return True

@nb.njit(nb.bool_(nb.float64[:],nb.float64[:],nb.float64[:]), cache=True)
def is_outside_container(position: np.ndarray,item_dim: np.ndarray,container_dim: np.ndarray):
    for k in range(3):
        if position[k]+item_dim[k]>container_dim[k]:
            return True
    return False

@nb.njit(nb.bool_[:](nb.float64[:,:],nb.float64[:],nb.float64[:]), cache=True)
def check_if_outside_containerv2(insertion_positions: np.ndarray,item_dim: np.ndarray,container_dim: np.ndarray):
    num_positions, _ = insertion_positions.shape
    is_out_of_container = np.zeros((num_positions,), dtype=np.bool_)
    for i in range(num_positions):
        for k in range(3):
            if insertion_positions[i,k]+item_dim[k]>container_dim[k]:
                is_out_of_container[i]=True
                break
    return is_out_of_container
    

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
                    dim_b: np.ndarray)->np.ndarray:
    n_a, _ = start_point_a.shape
    n_b, _ = start_point_b.shape
    end_a, end_b = start_point_a+dim_a, start_point_b+dim_b
    # is_intersect_all_dim = np.logical_not(np.logical_or(end_a[:, None, :]<=start_point_b[None, :, :], start_point_a[:, None, :]>=end_b[None, :, :]))
    is_intersect: np.ndarray = np.ones((n_a, n_b), dtype=np.bool_)
    for i in range(len(start_point_a)):
        for j in range(len(start_point_b)):
            for k in range(3):
                if end_a[i,k]<=start_point_b[j,k] or end_b[j,k]<=start_point_a[i,k]:
                    is_intersect[i,j]=False
                    break
                # if not is_intersect_all_dim[i,j,k]:
                #     is_intersect[i,j]=False
                # is_intersect[i,j] = np.all(is_intersect_all_dim[i,j,:])
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
    # intersection_length = np.maximum(intersection_length, 0.)
    retv2 = 1
    for k in range(2):
        if intersection_length[k]<=0:
            return 0
        retv2 *=intersection_length[k]
    return retv2

@nb.njit(nb.float64(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]), cache=True)
def compute_intersection_ndv2(start_point_a: np.ndarray,
                            dim_a: np.ndarray,
                            start_point_b: np.ndarray,
                            dim_b: np.ndarray,
                            end_b: np.ndarray)->float:
    end_a = start_point_a+dim_a
    intersection_start = np.maximum(start_point_a, start_point_b)
    intersection_end = np.minimum(end_a, end_b)
    intersection_length = intersection_end-intersection_start
    # intersection_length = np.maximum(intersection_length, 0.)
    retv2 = 1
    for k in range(2):
        if intersection_length[k]<=0:
            return 0
        retv2 *=intersection_length[k]
    return retv2


# @nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_intersect_nd_any_idx(ep: np.ndarray,
                        dummy_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray)->int:
    for i, inserted_item_dim in enumerate(inserted_item_dims):
        filled_position = filled_positions[i]
        # check if placing a very very small item here intersect with other inserted items
        if is_intersect_nd(ep, dummy_dim, filled_position, inserted_item_dim):
            return i
    return -1

@nb.njit(nb.bool(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_intersect_nd_any_v2(ep: np.ndarray,
                        item_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray)->bool:
    num_inserted_items, _ = inserted_item_dims.shape
    batch_size: int = 64
    num_batch: int = math.ceil(num_inserted_items/batch_size)
    for i in range(num_batch):
        ja:int = i*batch_size
        jb:int = min((i+1)*batch_size, num_inserted_items)
        if np.any(is_intersect_nd_vectorized(ep[None, :], item_dim[None, :], filled_positions[ja:jb], inserted_item_dims[ja:jb])):
            return True
    return False

@nb.njit(nb.bool[:](nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True)
def is_intersect_nd_any_vectorized(ext_points: np.ndarray,
                        item_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray,
                        inserted_item_end_points: np.ndarray)->np.ndarray:
    n_a, _ = ext_points.shape
    n_b, _ = inserted_item_dims.shape
    end_a, end_b = ext_points+item_dim[None, :], inserted_item_end_points
    is_intersect_any_items: np.ndarray = np.zeros((n_a,), dtype=np.bool_)
    for i in range(n_a):
        for j in range(n_b):
            intersection_occurs:bool = True
            for k in range(3):
                if end_a[i,k]<=filled_positions[j,k] or end_b[j,k]<=ext_points[i,k]:
                    intersection_occurs=False
                    break
            if intersection_occurs:
                is_intersect_any_items[i]=True
                break
    return is_intersect_any_items

@nb.njit(nb.bool[:](nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), cache=True,fastmath=True)
def is_intersect_nd_any_vectorized_ur(ext_points: np.ndarray,
                        item_dim: np.ndarray,
                        inserted_item_dims: np.ndarray,
                        filled_positions: np.ndarray,
                        inserted_item_end_points: np.ndarray)->np.ndarray:
    n_a, _ = ext_points.shape
    n_b, _ = inserted_item_dims.shape
    end_a, end_b = ext_points+item_dim[None, :], inserted_item_end_points
    is_intersect_any_items: np.ndarray = np.zeros((n_a,), dtype=np.bool_)
    for i in range(n_a):
        for j in range(n_b):
            if not (end_a[i,0]<=filled_positions[j,0] or end_b[j,0]<=ext_points[i,0] or\
                end_a[i,1]<=filled_positions[j,1] or end_b[j,1]<=ext_points[i,1] or\
                end_a[i,2]<=filled_positions[j,2] or end_b[j,2]<=ext_points[i,2]):
                is_intersect_any_items[i]=True
                break
    return is_intersect_any_items
        

@nb.njit(nb.int64(nb.int64,nb.int64,nb.float64,nb.float64), cache=True)
def binary_search_zone(start: int,
                       end: int,
                       q: float, 
                       zone_size: float)->int:
    while start <= end:
        mid: int = math.ceil((start+end)/2)
        z_a = mid*zone_size
        z_b = (mid+1)*zone_size
        if z_a <= q < z_b:
            return mid
        if q < z_a:
            end = mid-1
        else:
            start = mid+1
    return -1

@nb.njit(nb.int64[:](nb.float64[:],nb.float64[:],nb.float64[:],nb.float64), cache=True)
def get_item_zones(position: np.ndarray,
                   dim: np.ndarray,
                   container_dim: np.ndarray,
                   zone_size: float)->np.ndarray:
    n_zones = np.floor(container_dim/zone_size).astype(np.int64)
    # print(n_zones)
    s1 = binary_search_zone(0, n_zones[0], position[0], zone_size)
    e1 = binary_search_zone(s1, n_zones[0], position[0]+dim[0], zone_size)
    s2 = binary_search_zone(0, n_zones[1], position[1], zone_size)
    e2 = binary_search_zone(s2, n_zones[1], position[1]+dim[1], zone_size)
    s3 = binary_search_zone(0, n_zones[2], position[2], zone_size)
    e3 = binary_search_zone(s3, n_zones[2], position[2]+dim[2], zone_size)
    e1 = min(e1, n_zones[0]-1) 
    e2 = min(e2, n_zones[1]-1) 
    e3 = min(e3, n_zones[2]-1)
    # print(s1,s2,s3,e1,e2,e3)
    dx,dy,dz= e1-s1+1, e2-s2+1, e3-s3+1
    zones = np.arange(dx*dy*dz, dtype=np.int64)
    # print(zones)
    d2 = zones // (dy*dx)
    m2 = zones % (dy*dx)
    d1 = m2 // dx
    m1 = m2 % dx
    zones = d2*(n_zones[1]*n_zones[0]) + d1*n_zones[0] + m1
    # print(zones)
    zones += s1 + s2*n_zones[0] + s3*n_zones[0]*n_zones[1]
    # print(zones)
    # print("*********************")
    return zones
    