import json
import pathlib
import random
from typing import List, Tuple

import numpy as np
from haversine import haversine
import folium
import webbrowser

from problem.customer import Customer
from problem.hvrp3l import HVRP3L
from problem.item import Item
from problem.node import Node
from problem.vehicle import Vehicle

# ==== Node Class (simple) ====

class SimpleNode:
    def __init__(self, idx: int, coord: np.ndarray):
        self.idx = idx
        self.coord = coord

# ==== Item Generation ====

def classify_item_size(dim: np.ndarray) -> str:
    volume = np.prod(dim)
    if volume <= 1000:
        return "small"
    elif volume <= 10000:
        return "medium"
    else:
        return "large"

def sample_items_by_size(med_spec_dict, ratio: Tuple[float, float, float]) -> List[Item]:
    weight_threshold = random.uniform(20_000, 80_000)

    small_pool, medium_pool, large_pool = [], [], []
    for code, spec in med_spec_dict.items():
        try:
            dim = np.array([spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]], dtype=float)
            size = classify_item_size(dim)
            if size == "small":
                small_pool.append(code)
            elif size == "medium":
                medium_pool.append(code)
            else:
                large_pool.append(code)
        except Exception:
            continue

    size_pools = {"small": small_pool, "medium": medium_pool, "large": large_pool}
    items: List[Item] = []
    total_weight = 0.
    i = 0
    while total_weight < weight_threshold:
        size_choice = random.choices(["small", "medium", "large"], weights=ratio, k=1)[0]
        pool = size_pools[size_choice]
        if not pool:
            continue
        code = random.choice(pool)
        spec = med_spec_dict[code]
        try:
            weight = float(spec["berat_gram"])
            if total_weight + weight > weight_threshold:
                break
            l, w, h = spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]
            temp_req = spec["suhu_simpan"]
            dim = np.asarray([l, w, h], dtype=float)
            is_reefer_required = temp_req != "kamar"
            item = Item(i, code, dim, weight, False, is_reefer_required)
            items.append(item)
            total_weight += weight
            i += 1
        except Exception:
            continue
    return items

def get_customer_items_random_date(cust_id: str, ratio=(0.4, 0.4, 0.2)) -> List[Item]:
    med_spec_filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu_v2.json"
    with open(med_spec_filepath.absolute(), "r") as json_data:
        med_spec_dict = json.load(json_data)
    if not np.isclose(sum(ratio), 1.0):
        raise ValueError(f"Invalid ratio {ratio}: must sum to 1.0")
    return sample_items_by_size(med_spec_dict, ratio)

# ==== Vehicle Generation ====

def generate_vehicles(num_normal_trucks, num_reefer_trucks) -> List[Vehicle]:
    vehicle_filepath = pathlib.Path()/"raw_json"/"Kendaraan.json"
    with open(vehicle_filepath.absolute(), "r") as json_data:
        vec_dict = json.load(json_data)
    vehicles: List[Vehicle] = []
    i = 0
    normal_truck_dict = vec_dict["NORMAL_TRUCK"]
    for _ in range(num_normal_trucks):
        dim = np.asanyarray([normal_truck_dict["PANJANG"],
                             normal_truck_dict["LEBAR"],
                             normal_truck_dict["TINGGI"]], dtype=float)
        new_vec = Vehicle(i, normal_truck_dict["SERVICE"], normal_truck_dict["KAPASITAS"], dim,
                          False, normal_truck_dict["RATE"], normal_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1

    reefer_truck_dict = vec_dict["REEFER_TRUCK"]
    for _ in range(num_reefer_trucks):
        dim = np.asanyarray([reefer_truck_dict["PANJANG"],
                             reefer_truck_dict["LEBAR"],
                             reefer_truck_dict["TINGGI"]], dtype=float)
        new_vec = Vehicle(i, reefer_truck_dict["SERVICE"], reefer_truck_dict["KAPASITAS"], dim,
                          False, reefer_truck_dict["RATE"], reefer_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1
    return vehicles

# ==== Cluster and Coordinate Loader ====

def load_customers_and_depot(cabang: str):
    filepath = pathlib.Path()/"raw_json"/f"{cabang}.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    depot_coord = np.array([float(c) for c in data["CABANG"]["Maps"].split(",")], dtype=float)
    customers_data = data["CUSTOMERS"]
    customers = []
    for i, c in enumerate(customers_data):
        coord = np.array([float(x) for x in c["Maps"].split(",")])
        customers.append((coord, c["CUSTOMER_NUMBER"]))
    return depot_coord, customers

def get_far_centers(data, num_clusters, min_distance_km):
    centers = []
    remaining_data = data.copy()
    while len(centers) < num_clusters and remaining_data:
        new_center = remaining_data[np.random.choice(len(remaining_data))]
        centers.append(new_center)
        distances = [haversine(new_center[0], d[0], unit='km') for d in remaining_data]
        remaining_data = [d for d, dist in zip(remaining_data, distances) if dist >= min_distance_km]
    return centers

def build_clusters(all_data, centers, points_per_cluster=10, max_radius_km=5):
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

# ==== Main Entry: Generate Problem ====

def generate_problem(cabang: str,
                     num_clusters: int = 4,
                     points_per_cluster: int = 10,
                     item_ratio: Tuple[float, float, float] = (0.6, 0.3, 0.1),
                     num_normal_trucks: int = 10,
                     num_reefer_trucks: int = 10):
    depot_coord, all_customers = load_customers_and_depot(cabang)
    centers = get_far_centers(all_customers, num_clusters=num_clusters, min_distance_km=15)
    clusters = build_clusters(all_customers, centers, points_per_cluster=points_per_cluster, max_radius_km=5)

    selected_customers = [cust for cluster in clusters for cust in cluster]
    customers: List[Customer] = []
    for i, (coord, cust_id) in enumerate(selected_customers):
        items = get_customer_items_random_date(str(cust_id), ratio=item_ratio)
        print(f"[Customer {cust_id}] Items generated: {len(items)} | Total weight: {sum(item.weight for item in items):.1f}g")
        new_cust = Customer(i+1, cust_id, coord, items)
        customers.append(new_cust)


    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(cabang)

    # Optionally: visualize
    map = folium.Map(tuple(depot_coord), zoom_start=12)
    for coord, _ in selected_customers:
        folium.Marker(location=[coord[0], coord[1]]).add_to(map)
    map.save("map.html")
    webbrowser.open("map.html")

if __name__ == "__main__":
    generate_problem("JK2")
