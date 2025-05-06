import argparse
import json
import pathlib
import random
from typing import List, Optional, Tuple

import numpy as np
from ep_heuristic.random_slpack import try_slpack
from haversine import haversine
from problem.customer import Customer
from problem.hvrp3l import HVRP3L
from problem.item import POSSIBLE_ROTATION_PERMUTATION_MATS, Item
from problem.node import Node
from problem.vehicle import Vehicle

VOLUME_THRESHOLD = 500000

def parse_args():
    parser = argparse.ArgumentParser(description="instance generation args.")
    parser.add_argument("--region",
                        type=str,
                        required=True,
                        choices=["JK2","SBY","MKS"],
                        help="region for depot choice")
    parser.add_argument("--num-customers",
                        type=int,
                        required=True,
                        help="number of customers")
    parser.add_argument("--small-items-ratio",
                        type=float,
                        help="ratio of small items in customer orders")
    parser.add_argument("--large-items-ratio",
                        type=float,
                        help="ratio of large items in customer orders")
    parser.add_argument("--num-reefer-trucks",
                        type=int,
                        required=True,
                        help="num of reefer trucks")
    parser.add_argument("--num-normal-trucks",
                        type=int,
                        required=True,
                        help="num of normal trucks")
    
    parser.add_argument("--num-clusters",
                        type=int,
                        default=1,
                        help="num of clusters, 1=no cluster/random")
    parser.add_argument("--demand-mode",
                        type=str,
                        choices=["historical","generated"],
                        required=True,
                        help="historical = sampling from historical demand, generated=sampling from items")
    
    
    return parser.parse_args()



def get_maximum_packable_items(items:List[Item])->List[Item]:
    packable_items: List[Item] = []
    vehicles = generate_vehicles(1,1)
    vehicle = vehicles[1]
    if np.any(np.asanyarray([item.is_reefer_required for item in items])):
        vehicle = vehicles[0]

    total_volume = 0
    total_weight = 0
    for item in items:
        if total_volume + item.volume > vehicle.volume_capacity or total_weight + item.weight > vehicle.weight_capacity:
            continue 
        packable_items.append(item)
        total_num_items = len(packable_items)
        item_dims = np.stack([item_.dim for item_ in packable_items])
        item_priorities = np.zeros((total_num_items,), dtype=int)
        sorted_idx = np.arange(total_num_items)
        rotation_trial_idx = np.zeros((total_num_items, 2), dtype=int)
        rotation_trial_idx[:, 1] = 1

        _, _, is_packing_feasible = try_slpack(item_dims, item_priorities, sorted_idx, rotation_trial_idx, vehicle.container_dim, POSSIBLE_ROTATION_PERMUTATION_MATS, 0.8, 5)
        if not is_packing_feasible:
            packable_items.pop()
    return packable_items

def get_customer_items_random_date(cust_id: str)->List[Item]:
    transaction_filepath = pathlib.Path()/"raw_json"/"Transaksi_v2.json"
    with open(transaction_filepath.absolute(), "r") as json_data:
        trans_dict = json.load(json_data)
    med_spec_filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu_v2.json"
    with open(med_spec_filepath.absolute(), "r") as json_data:
        med_spec_dict = json.load(json_data)
    
    items: List[Item] = []
    try:
        historical_transaction_dict = trans_dict[cust_id]
        r_date = random.choice(list(historical_transaction_dict.keys()))
        raw_item_dict_list = historical_transaction_dict[r_date]
        i = 0
        date_items: List[Item] = []
        for raw_item_dict in raw_item_dict_list:
            product_code = raw_item_dict["PRODUCT_CODE"]
            qty = raw_item_dict["SHIPPED_QTY"]
            med_spec = med_spec_dict[product_code]
            weight = float(med_spec["berat_gram"])
            l,w,h = med_spec["panjang_cm"], med_spec["lebar_cm"], med_spec["tinggi_cm"]
            temp_req = med_spec["suhu_simpan"]
            dim = np.asanyarray([l,w,h], dtype=float)
            if np.any(dim>150):
                continue
            is_reefer_required = temp_req != "kamar"
            for _ in range(qty):
                new_item = Item(i, product_code, dim, weight, False, is_reefer_required)
                date_items.append(new_item)
                i += 1
        
        all_items_total_volume = sum(item.volume for item in date_items)
        if all_items_total_volume > VOLUME_THRESHOLD:
            random.shuffle(date_items)
            total_volume = 0
            for item in date_items:
                total_volume += item.volume
                if total_volume > VOLUME_THRESHOLD:
                    break
                items.append(item)
        else:
            items = date_items
            
        
    except KeyError:
        # generate items randomly
        threshold = 0.05*5000000 #gram
        total_weight = 0.
        total_volume = 0
        i = 0
        while total_weight < threshold and total_volume<VOLUME_THRESHOLD:
            qty = random.randint(1, 10)
            product_code = random.choice(list(med_spec_dict.keys()))
            med_spec = med_spec_dict[product_code]
            weight = float(med_spec["berat_gram"])
            l,w,h = med_spec["panjang_cm"], med_spec["lebar_cm"], med_spec["tinggi_cm"]
            temp_req = med_spec["suhu_simpan"]
            dim = np.asanyarray([l,w,h], dtype=float)
            if np.any(dim>150):
                continue
            volume = dim.prod()
            is_reefer_required = temp_req != "kamar"
            for _ in range(qty):
                total_weight += weight
                total_volume += volume
                if total_weight > threshold or total_volume > VOLUME_THRESHOLD:
                    break
                new_item = Item(i, product_code, dim, weight, False, is_reefer_required)
                items.append(new_item)
                i += 1

    return items


def classify_item_size(dim: np.ndarray) -> str:
    volume = np.prod(dim)
    if volume <= 1000:
        return "small"
    elif volume <= 10000:
        return "medium"
    else:
        return "large"

def sample_items_by_size(med_spec_dict, ratio: Tuple[float, float, float]) -> List[Item]:
    weight_threshold = random.uniform(50_000, 80_000)
    small_pool, medium_pool, large_pool = [], [], []
    for code, spec in med_spec_dict.items():
        dim = np.array([spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]], dtype=float)
        if np.any(dim>150):
            continue
        size = classify_item_size(dim)
        if size == "small":
            small_pool.append(code)
        elif size == "medium":
            medium_pool.append(code)
        else:
            large_pool.append(code)

    size_pools = {"small": small_pool, "medium": medium_pool, "large": large_pool}
    items: List[Item] = []
    total_weight = 0.
    total_volume = 0.
    i = 0
    while total_weight < weight_threshold and total_volume < VOLUME_THRESHOLD:
        size_choice = random.choices(["small", "medium", "large"], weights=ratio, k=1)[0]
        pool = size_pools[size_choice]
        if not pool:
            continue
        code = random.choice(pool)
        spec = med_spec_dict[code]
        weight = float(spec["berat_gram"])
        if total_weight + weight > weight_threshold:
            break
        l, w, h = spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]
        temp_req = spec["suhu_simpan"]
        dim = np.asarray([l, w, h], dtype=float)
        is_reefer_required = temp_req != "kamar"
        item = Item(i, code, dim, weight, False, is_reefer_required)
        total_volume += item.volume
        total_weight += weight
        # if total_weight >= weight_threshold or total_volume >= VOLUME_THRESHOLD:
        #     break
        items.append(item)
        i += 1
    return items

def generate_items_by_ratio(ratio) -> List[Item]:
    med_spec_filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu_v2.json"
    with open(med_spec_filepath.absolute(), "r") as json_data:
        med_spec_dict = json.load(json_data)
    if not np.isclose(sum(ratio), 1.0):
        raise ValueError(f"Invalid ratio {ratio}: must sum to 1.0")
    return sample_items_by_size(med_spec_dict, ratio)

# ==== Cluster and Coordinate Loader ====

def load_customers_and_depot(cabang: str):
    filepath = pathlib.Path()/"raw_json"/f"{cabang}.json"
    with open(filepath, "r", encoding="utf-8") as f:
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



def generate_customers(cabang:str, 
                       num_customers:int, 
                       num_clusters:int, 
                       demand_mode:int,
                       ratio: Optional[Tuple[float,float,float]]=None)->List[Customer]:
    customers: List[Customer] = []
    depot_coord, all_customers = load_customers_and_depot(cabang)
    points_per_cluster = num_customers//num_clusters

    coords_cust_id_list = []
    if num_clusters == 1:
        random_custs = random.sample(all_customers, num_customers)
        coords_cust_id_list = [(cust[0], cust[1]) for cust in random_custs]
    else:
        centers = get_far_centers(all_customers, num_clusters=num_clusters, min_distance_km=15)
        clusters = build_clusters(all_customers, centers, points_per_cluster=points_per_cluster, max_radius_km=5)
        coords_cust_id_list = [cust for cluster in clusters for cust in cluster]
    
    for i, (coord, cust_id) in enumerate(coords_cust_id_list):
        items: List[Item] = []
        while len(items)==0:
            if demand_mode == "historical":
                items = get_customer_items_random_date(str(cust_id))
                print(f"[Customer {cust_id}] Items generated: {len(items)} | Total weight: {sum(item.weight for item in items):.1f}g | Total volume: {sum(item.volume for item in items)}")
            else:
                if ratio is None:
                    raise ValueError("if demand mode is not historical (generated), item size ratio must be provided")
                items = generate_items_by_ratio(ratio)
                print(f"[{ratio} Customer {cust_id}] Items generated: {len(items)} | Total weight: {sum(item.weight for item in items):.1f}g | Total volume: {sum(item.volume for item in items)}")
            items = get_maximum_packable_items(items)
        new_cust = Customer(i+1, cust_id, coord, items)
        customers.append(new_cust)

    return customers
     
def generate(cabang: str, 
             num_customers: int,
             demand_mode: str,
             num_clusters: int,
             num_normal_trucks:int,
             num_reefer_trucks:int,
             ratio:Optional[Tuple[float,float,float]]):
    filename = cabang+".json"
    filepath = pathlib.Path()/"raw_json"/filename
    customers: List[Node] = []
    with open(filepath.absolute(), "r", encoding="utf-8") as json_data:
        d = json.load(json_data)
        
    depot_coord = d["CABANG"]["Maps"]
    depot_coord = depot_coord.split(",")
    depot_coord = np.asanyarray([float(c) for c in depot_coord], dtype=float)
    
    customers = generate_customers(cabang, num_customers, num_clusters, demand_mode, ratio)


    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(cabang, demand_mode, num_clusters, ratio)
    

def generate_vehicles(num_normal_trucks, num_reefer_trucks)->List[Vehicle]:
    vehicle_filepath = pathlib.Path()/"raw_json"/"Kendaraan.json"
    with open(vehicle_filepath.absolute(), "r") as json_data:
        vec_dict = json.load(json_data)
    vehicles: List[Vehicle] = []
    i = 0
    reefer_truck_dict = vec_dict["REEFER_TRUCK"]
    for _ in range(num_reefer_trucks):
        l,w,h = reefer_truck_dict["PANJANG"], reefer_truck_dict["LEBAR"], reefer_truck_dict["TINGGI"]
        dim = np.asanyarray([l,w,h], dtype=float)
        new_vec = Vehicle(i,
                          reefer_truck_dict["SERVICE"],
                          reefer_truck_dict["KAPASITAS"],
                          dim,
                          True,
                          reefer_truck_dict["RATE"],
                          reefer_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1
    normal_truck_dict = vec_dict["NORMAL_TRUCK"]
    for _ in range(num_normal_trucks):
        l,w,h = normal_truck_dict["PANJANG"], normal_truck_dict["LEBAR"], normal_truck_dict["TINGGI"]
        dim = np.asanyarray([l,w,h], dtype=float)
        new_vec = Vehicle(i,
                          normal_truck_dict["SERVICE"],
                          normal_truck_dict["KAPASITAS"],
                          dim,
                          False,
                          normal_truck_dict["RATE"],
                          normal_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1
    
    
    return vehicles

if __name__=="__main__":
    args = parse_args()
    ratio = None
    if args.demand_mode == "generated":
        s = args.small_items_ratio
        l = args.large_items_ratio
        m = 1 - s - l
        ratio = np.asanyarray([s,m,l])
        ratio /= ratio.sum()
        ratio = tuple(ratio.tolist())

    generate(args.region,
             args.num_customers,
             args.demand_mode,
             args.num_clusters,
             args.num_customers,
             args.num_customers,
             ratio)