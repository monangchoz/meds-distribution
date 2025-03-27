import json
import pathlib
from typing import List

import numpy as np
import matplotlib.pyplot as plt

#Import Map Visualization
import folium
import webbrowser


class Node:
    def __init__(self, idx: int, coord: np.ndarray):
        self.idx = idx
        self.coord = coord
        
class Customer(Node):
    def __init__(self, idx, coord, cust_id: int):
        super().__init__(idx, coord)
        self.cust_id = cust_id
        
        
def generate(cabang: str, num_customers: int):
    filename = cabang+".json"
    filepath = pathlib.Path()/"raw_json"/filename
    depot: Node
    customers: List[Node] = []
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)
        
    depot_coord = d["CABANG"]["Maps"]
    depot_coord = depot_coord.split(",")
    depot_coord = np.asanyarray([float(c) for c in depot_coord], dtype=float)
    depot = Node(0, depot_coord)
    
    customers_dict_list = d["CUSTOMERS"]
    chosen_customers_idx = np.random.choice(len(customers_dict_list), size=num_customers, replace=False)
    for i, c_idx in enumerate(chosen_customers_idx):
        customer_dict = customers_dict_list[c_idx]
        c_coord = customer_dict["Maps"].split(",")
        c_coord = np.asanyarray([float(c) for c in c_coord], dtype=float)
        cust_id = customer_dict["CUSTOMER_NUMBER"]
        new_cust = Customer(i+1, c_coord, cust_id)
        customers += [new_cust]
    cust_coords = np.stack([customer.coord for customer in customers])

    #Add maps display?
    map = folium.Map((depot_coord[0],depot_coord[1]),zoom_start=12)

    #print(cust_coords)

    
    for i in range (len(chosen_customers_idx)):
        print(cust_coords[i])
        folium.Marker(location=[cust_coords[i][0],cust_coords[i][1]]).add_to(map)   

    map.save("map.html")
    webbrowser.open("map.html")

    '''
    plt.scatter(cust_coords[:, 0], cust_coords[:, 1])
    plt.scatter(depot_coord[0],depot_coord[1])
    plt.show()
    '''    


if __name__=="__main__":
    generate("JK2", 433)