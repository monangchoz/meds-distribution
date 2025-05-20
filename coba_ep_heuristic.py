import numpy as np

from ep_heuristic.random_slpack import random_slpack
from utils import visualize_container


def run():
    num_items = 10
    item_dims = np.random.randint(1, 5, size=(num_items, 3)).astype(float)
    item_weights = np.random.randint(1,5, size=(num_items,))
    item_volumes = np.prod(item_dims, axis=1)
    item_priorities = np.arange(num_items, dtype=int)
    
    container_dim = np.asanyarray([8,6,6], dtype=float)
    container_weight_cap = 1000
    container_volume_cap = np.prod(container_dim)
    
    
    positions, rotations, is_feasible = random_slpack(item_dims, item_volumes, item_priorities, container_dim, base_support_alpha=0.8, max_trial=100)
    if not is_feasible:
        exit()
    actual_dims = item_dims[np.arange(num_items)[:, None], rotations]
    visualize_container(container_dim, positions, actual_dims)

if __name__ == "__main__":
    run()