from ep_heuristic.insertion import insert_items

import numpy as np

def run():
    num_items = 3
    item_dims = np.random.randint(1, 5, size=(num_items, 3)).astype(float)
    item_weights = np.random.randint(1,5, size=(num_items,))
    item_volumes = np.prod(item_dims, axis=1)
    item_priorities = np.arange(num_items)
    
    container_dim = np.asanyarray([10,10,10], dtype=float)
    container_weight_cap = 1000
    container_volume_cap = np.prod(container_dim)
    
    
    res = insert_items(item_dims, container_dim)
    print(res)
if __name__ == "__main__":
    run()