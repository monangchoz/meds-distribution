import numpy as np

from ep_heuristic.insertion import insert_items
from utils import visualize_container


def run():
    num_items = 8
    item_dims = np.random.randint(1, 5, size=(num_items, 3)).astype(float)
    item_weights = np.random.randint(1,5, size=(num_items,))
    item_volumes = np.prod(item_dims, axis=1)
    item_priorities = np.arange(num_items)
    
    container_dim = np.asanyarray([5,6,6], dtype=float)
    container_weight_cap = 1000
    container_volume_cap = np.prod(container_dim)
    
    
    positions, rotations, is_feasible = insert_items(item_dims, container_dim)
    if not is_feasible:
        exit()
    actual_dims = item_dims[np.arange(num_items)[:, None], rotations]
    visualize_container(container_dim, positions, actual_dims)

if __name__ == "__main__":
    run()