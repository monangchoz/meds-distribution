import json
import pathlib
import random
from typing import List, Tuple

import numpy as np

from problem.customer import Customer
from problem.hvrp3l import HVRP3L
from problem.item import Item
from problem.node import Node
from problem.vehicle import Vehicle        

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

    """
    Randomly generates items from the product catalog until total weight exceeds threshold.
    Item types are sampled according to the (small, medium, large) ratio.
    """
    # Prepare product pools
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

    size_pools = {
        "small": small_pool,
        "medium": medium_pool,
        "large": large_pool
    }

    ratio_map = {
        "small": ratio[0],
        "medium": ratio[1],
        "large": ratio[2]
    }

    items: List[Item] = []
    total_weight = 0.
    i = 0
    category_counter = {"small": 0, "medium": 0, "large": 0}

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
            category_counter[size_choice] += 1
            i += 1
        except Exception:
            continue

    print("Item generation complete:")
    print(f"  Total weight: {total_weight:.2f} g")
    print(f"  Total items: {len(items)}")
    for k, v in category_counter.items():
        print(f"  {k.capitalize()}: {v} items")

    return items

def get_customer_items_random_date(cust_id: str, ratio=(0.4, 0.4, 0.2)) -> List[Item]:
    """
    Always generates random items for a customer using product specs only (no historical data).
    """
    med_spec_filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu_v2.json"
    with open(med_spec_filepath.absolute(), "r") as json_data:
        med_spec_dict = json.load(json_data)

    if not np.isclose(sum(ratio), 1.0):
        raise ValueError(f"Invalid ratio {ratio}: must sum to 1.0")

    items = sample_items_by_size(med_spec_dict, ratio)
    return items

def generate_customers(cabang, num_customers, ratio=(0.4, 0.4, 0.2)) -> List[Customer]:
    filename = cabang+".json"
    filepath = pathlib.Path()/"raw_json"/filename
    customers: List[Customer] = []
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)

    customers_dict_list = d["CUSTOMERS"]
    chosen_customers_idx = np.random.choice(len(customers_dict_list), size=num_customers, replace=False)
    for i, c_idx in enumerate(chosen_customers_idx):
        customer_dict = customers_dict_list[c_idx]
        c_coord = customer_dict["Maps"].split(",")
        c_coord = np.asanyarray([float(c) for c in c_coord], dtype=float)
        cust_id = customer_dict["CUSTOMER_NUMBER"]
        items = get_customer_items_random_date(str(cust_id), ratio)
        new_cust = Customer(i+1, cust_id, c_coord, items)
        customers += [new_cust]
    return customers

def generate(cabang: str, num_customers: int,
             num_normal_trucks: int = 10,
             num_reefer_trucks: int = 10,
             ratio: Tuple[float, float, float] = (0.4, 0.4, 0.2)):
    filename = cabang+".json"
    filepath = pathlib.Path()/"raw_json"/filename
    customers: List[Node] = []
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)

    depot_coord = d["CABANG"]["Maps"]
    depot_coord = depot_coord.split(",")
    depot_coord = np.asanyarray([float(c) for c in depot_coord], dtype=float)

    customers = generate_customers(cabang, num_customers, ratio)
    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(cabang)

def generate_vehicles(num_normal_trucks, num_reefer_trucks) -> List[Vehicle]:
    vehicle_filepath = pathlib.Path()/"raw_json"/"Kendaraan.json"
    with open(vehicle_filepath.absolute(), "r") as json_data:
        vec_dict = json.load(json_data)
    vehicles: List[Vehicle] = []
    i = 0
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

    reefer_truck_dict = vec_dict["REEFER_TRUCK"]
    for _ in range(num_reefer_trucks):
        l,w,h = reefer_truck_dict["PANJANG"], reefer_truck_dict["LEBAR"], reefer_truck_dict["TINGGI"]
        dim = np.asanyarray([l,w,h], dtype=float)
        new_vec = Vehicle(i,
                          reefer_truck_dict["SERVICE"],
                          reefer_truck_dict["KAPASITAS"],
                          dim,
                          False,
                          reefer_truck_dict["RATE"],
                          reefer_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1
    return vehicles

if __name__=="__main__":
    generate("JK2", 30, ratio=(0.8, 0.15, 0.05))
