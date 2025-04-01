import numpy as np

@np.vectorize(otypes=[bool])
def is_intersect_1d(start_point_a: float, 
                    length_a: float,
                    start_point_b: float, 
                    length_b: float)->bool:
    end_a, end_b = start_point_a+length_a, start_point_b+length_b
    return not (end_a <= start_point_b or end_b <= start_point_a)

def is_intersect_nd(start_point_a: np.ndarray,
                    dim_a: np.ndarray,
                    start_point_b: np.ndarray,
                    dim_b: np.ndarray)->np.bool:
    # this can be turned into default numpy vectorized operations, later.
    return np.all(is_intersect_1d(start_point_a, dim_a, start_point_b, dim_b))

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

def compute_intersection_nd(start_point_a: np.ndarray,
                            dim_a: np.ndarray,
                            start_point_b: np.ndarray,
                            dim_b: np.ndarray)->float:
    # this can be turned into default numpy vectorized operations, later.
    return np.prod(compute_intersection_1d(start_point_a, dim_a, start_point_b, dim_b))