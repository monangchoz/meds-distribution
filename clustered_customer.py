import json
import pathlib
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine

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

#1. Generate Coordinates from JSON 

def generate(cabang: str):
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
    all_customers_idx =  list(range(len(customers_dict_list)))
    for i, c_idx in enumerate(all_customers_idx):
        customer_dict = customers_dict_list[c_idx]
        c_coord = customer_dict["Maps"].split(",")
        c_coord = np.asanyarray([float(c) for c in c_coord], dtype=float)
        cust_id = customer_dict["CUSTOMER_NUMBER"]
        new_cust = Customer(i+1, c_coord, cust_id)
        customers += [new_cust]
    cust_coords = np.stack([customer.coord for customer in customers])

    return cust_coords

#2 Select Cluster Centers

def get_far_centers(coords, num_clusters, min_distance_km):
    centers = []
    remaining_coords = coords.copy()
    
    while len(centers) < num_clusters and len(remaining_coords) > 0:
        # Pick a random center from remaining coordinates
        new_center = remaining_coords[np.random.choice(len(remaining_coords))]
        centers.append(new_center)
        
        # Remove all coords within min_distance_km of the new center
        distances = [haversine(new_center, coord, unit='km') for coord in remaining_coords]
        remaining_coords = remaining_coords[np.array(distances) >= min_distance_km]
        
    return np.array(centers)

#3 Build Cluster

def build_clusters(all_coords, centers, points_per_cluster=20, max_radius_km=5):
    clusters = []
    remaining_coords = all_coords.copy()
    
    for center in centers:
        # Calculate distances to the current center
        distances = np.array([haversine(center, coord, unit='km') for coord in remaining_coords])
        
        # Select the closest (points_per_cluster) points within max_radius_km
        eligible_indices = np.where(distances <= max_radius_km)[0]

        # Add to fill the remaining spot on the cluster if the coordinates in the cluster is not enough
        if len(eligible_indices) < points_per_cluster:
            selected_indices = np.argsort(distances)[:points_per_cluster]
            cluster_points = remaining_coords[selected_indices]
            
            clusters.append(cluster_points)
            
            # Remove selected points from remaining_coords
            remaining_coords = np.delete(remaining_coords, selected_indices, axis=0)
        else:
            selected_indices = np.argsort(distances[eligible_indices])[:points_per_cluster]
            cluster_points = remaining_coords[eligible_indices][selected_indices]
            
            clusters.append(cluster_points)
            
            # Remove selected points from remaining_coords
            remaining_coords = np.delete(remaining_coords, eligible_indices[selected_indices], axis=0)

        # print(eligible_indices)
    return clusters

# ===== 4. PRINT RESULTS =====
def print_results ():
    for i, (center, cluster) in enumerate(zip(cluster_centers, clusters)):
        print(f"Cluster {i+1}: Center = {center}")
        print(f"Points ({len(cluster)}):")
        print(cluster)
        print(f"Max distance from center: {max([haversine(center, p, unit='km') for p in cluster]):.2f} km\n")


if __name__=="__main__":
    all_coordinates = generate("JK2")
    cluster_centers = get_far_centers(all_coordinates, num_clusters=2, min_distance_km=15)
    clusters = build_clusters(all_coordinates, cluster_centers, points_per_cluster=20, max_radius_km=10)
    print_results()
    #print(clusters[0][0])

    map = folium.Map((-6.200000, 106.750000),zoom_start=12)
    for i in range (len(clusters)):
        for j in range(len(clusters[i])):
            #print(clusters[i][j])
            folium.Marker(location=[clusters[i][j][0],clusters[i][j][1]]).add_to(map)  

    map.save("map.html")
    webbrowser.open("map.html")
