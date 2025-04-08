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


def get_product_size_categories() -> Tuple[set, set, set]:
    """Returns sets of product codes categorized into small, medium, and large."""
    med_spec_filepath = pathlib.Path() / "raw_json" / "Dimensi_Suhu_v2.json"
    with open(med_spec_filepath.absolute(), "r") as f:
        med_spec_dict = json.load(f)

    product_data = []
    for product_code, spec in med_spec_dict.items():
        try:
            l = float(spec["panjang_cm"])
            w = float(spec["lebar_cm"])
            h = float(spec["tinggi_cm"])
            volume = l * w * h
            product_data.append((product_code, volume))
        except:
            continue

    volumes = np.array([v for _, v in product_data])
    q1 = np.quantile(volumes, 0.33)
    q2 = np.quantile(volumes, 0.66)

    small = {code for code, vol in product_data if vol <= q1}
    medium = {code for code, vol in product_data if q1 < vol <= q2}
    large = {code for code, vol in product_data if vol > q2}

    return small, medium, large


def get_customer_items_by_filter(cust_id: str, allowed_codes: set) -> List[Item]:
    transaction_path = pathlib.Path() / "raw_json" / "Transaksi_v2.json"
    spec_path = pathlib.Path() / "raw_json" / "Dimensi_Suhu_v2.json"

    with open(transaction_path.absolute(), "r") as f:
        trans_dict = json.load(f)
    with open(spec_path.absolute(), "r") as f:
        med_spec_dict = json.load(f)

    items: List[Item] = []
    try:
        transaction = trans_dict[cust_id]
        r_date = random.choice(list(transaction.keys()))
        raw_items = transaction[r_date]

        if not all(item["PRODUCT_CODE"] in allowed_codes for item in raw_items):
            return []

        i = 0
        for raw in raw_items:
            product_code = raw["PRODUCT_CODE"]
            qty = raw["SHIPPED_QTY"]
            spec = med_spec_dict[product_code]
            weight = float(spec["berat_gram"])
            l, w, h = spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]
            dim = np.array([l, w, h], dtype=float)
            temp_req = spec["suhu_simpan"]
            is_reefer = temp_req != "kamar"

            for _ in range(qty):
                items.append(Item(i, product_code, dim, weight, False, is_reefer))
                i += 1
    except:
        return []

    return items


def get_customers_by_category(cabang: str, allowed_codes: set, count: int) -> List[Customer]:
    filepath = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(filepath.absolute(), "r") as f:
        branch_data = json.load(f)

    customers_data = branch_data["CUSTOMERS"]
    random.shuffle(customers_data)

    customers = []
    i = 0
    for cust in customers_data:
        if len(customers) >= count:
            break
        coord = np.array([float(x) for x in cust["Maps"].split(",")])
        cust_id = cust["CUSTOMER_NUMBER"]
        items = get_customer_items_by_filter(str(cust_id), allowed_codes)
        if items:
            customers.append(Customer(i + 1, cust_id, coord, items))
            i += 1
    return customers


def generate_vehicles(num_normal_trucks, num_reefer_trucks) -> List[Vehicle]:
    filepath = pathlib.Path() / "raw_json" / "Kendaraan.json"
    with open(filepath.absolute(), "r") as f:
        vdata = json.load(f)

    vehicles: List[Vehicle] = []
    i = 0

    for _ in range(num_normal_trucks):
        v = vdata["NORMAL_TRUCK"]
        dim = np.array([v["PANJANG"], v["LEBAR"], v["TINGGI"]], dtype=float)
        vehicles.append(Vehicle(i, v["SERVICE"], v["KAPASITAS"], dim, False, v["RATE"], v["VARIABLE_RATE"]))
        i += 1

    for _ in range(num_reefer_trucks):
        v = vdata["REEFER_TRUCK"]
        dim = np.array([v["PANJANG"], v["LEBAR"], v["TINGGI"]], dtype=float)
        vehicles.append(Vehicle(i, v["SERVICE"], v["KAPASITAS"], dim, False, v["RATE"], v["VARIABLE_RATE"]))
        i += 1

    return vehicles


def generate(cabang: str,
             total_customers: int,
             ratio_small: float,
             ratio_medium: float,
             ratio_large: float,
             num_normal_trucks: int = 10,
             num_reefer_trucks: int = 10):

    assert abs(ratio_small + ratio_medium + ratio_large - 1.0) < 0.01, "Ratios must sum to 1.0"

    small_codes, medium_codes, large_codes = get_product_size_categories()

    n_small = round(total_customers * ratio_small)
    n_medium = round(total_customers * ratio_medium)
    n_large = total_customers - n_small - n_medium  # Ensure total matches exactly

    small_customers = get_customers_by_category(cabang, small_codes, n_small)
    medium_customers = get_customers_by_category(cabang, medium_codes, n_medium)
    large_customers = get_customers_by_category(cabang, large_codes, n_large)

    all_customers = small_customers + medium_customers + large_customers
    random.shuffle(all_customers)

    filepath = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(filepath.absolute(), "r") as f:
        branch_data = json.load(f)
    depot_coord = np.array([float(x) for x in branch_data["CABANG"]["Maps"].split(",")])

    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)

    problem = HVRP3L(depot_coord, all_customers, vehicles)
    problem.to_json(f"{cabang}_mix_{int(ratio_small*100)}s_{int(ratio_medium*100)}m_{int(ratio_large*100)}l")


if __name__ == "__main__":
    # Example: 20% small, 30% medium, 50% large customers from branch JK2
    generate("JK2", 100, ratio_small=0.2, ratio_medium=0.3, ratio_large=0.5)
