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
    cust_dict = np.stack([{"id": customer.cust_id, "coord": customer.coord} for customer in customers])
    # print(cust_dict)
    return cust_dict

#2 Select Cluster Centers

def get_far_centers(data, num_clusters, min_distance_km):
    centers = []
    remaining_data = data.copy()

    while len(centers) < num_clusters and len(remaining_data) > 0:
        new_center = remaining_data[np.random.choice(len(remaining_data))]
        centers.append(new_center)

        distances = [haversine(new_center[0], d[0], unit='km') for d in remaining_data]
        remaining_data = [d for d, dist in zip(remaining_data, distances) if dist >= min_distance_km]

    return centers

#3 Build Cluster

def build_clusters(all_data, centers, points_per_cluster=20, max_radius_km=5):
    clusters = []
    remaining_data = all_data.copy()

    for center in centers:
        distances = np.array([haversine(center[0], d[0], unit='km') for d in remaining_data])

        eligible_indices = np.where(distances <= max_radius_km)[0]

        if len(eligible_indices) < points_per_cluster:
            selected_indices = np.argsort(distances)[:points_per_cluster]
        else:
            selected_indices = eligible_indices[np.argsort(distances[eligible_indices])[:points_per_cluster]]

        cluster_points = [remaining_data[i] for i in selected_indices]
        clusters.append(cluster_points)

        remaining_data = [d for i, d in enumerate(remaining_data) if i not in selected_indices]

    return clusters

# ===== 4. PRINT RESULTS =====
def print_results ():
    for i, (center, cluster) in enumerate(zip(cluster_centers, clusters)):
        print(f"Cluster {i+1}: Center = {center}")
        print(f"Points ({len(cluster)}):")
        print(cluster)
        print(f"Max distance from center: {max([haversine(center, p, unit='km') for p in cluster]):.2f} km\n")


# 5. Buat random pesanan pelanggan berdasarkan tanggal pembelian.

def random_orders(ids, filepath: str, output_path: str = None):
    with open(filepath, "r") as f:
        transaction_data = json.load(f)

    ids_set = set(str(i) for i in ids)
    
    randomized_data = {}

    for cust_id, date_dict in transaction_data.items():

        if cust_id not in ids_set:
            continue
        dates = list(date_dict.keys())
        chosen_date = np.random.choice(dates)
        transactions = date_dict[chosen_date]

        randomized_data[cust_id] = {chosen_date: transactions}

    print(randomized_data)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(randomized_data, f)

    return randomized_data

if __name__=="__main__":
    all_coordinates_codes = generate("JK2")

    combined_data = [(item["coord"], item["id"]) for item in all_coordinates_codes]

    cluster_centers = get_far_centers(combined_data, num_clusters=4, min_distance_km=15)
    clusters = build_clusters(combined_data, cluster_centers, points_per_cluster=10, max_radius_km=5)

    # for idx, cluster in enumerate(clusters):
    #     print(f"Cluster {idx+1}:")
    #     for coord, cid in cluster:
    #         print(f"ID: {cid}, Coord: {coord}")

    filtered_ids = [cid for cluster in clusters for _, cid in cluster]

    print(len(filtered_ids))

    random_orders(filtered_ids, pathlib.Path()/"raw_json"/"Transaksi_v2.json", pathlib.Path()/"raw_json"/"orders_output.json")
    map = folium.Map((-6.200000, 106.750000),zoom_start=12)
    for cluster in clusters:
        for coord, _ in cluster:
            folium.Marker(location=[coord[0], coord[1]]).add_to(map)  

    map.save("map.html")
    webbrowser.open("map.html")


