import json
import pathlib
import random
from typing import List

import numpy as np

from problem.customer import Customer
from problem.hvrp3l import HVRP3L
from problem.item import Item
from problem.node import Node
from problem.vehicle import Vehicle


def get_product_codes_by_category(category: str) -> set:
    assert category in {"small", "medium", "large"}, "Invalid category"

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
        except Exception:
            continue

    volumes = np.array([v for _, v in product_data])
    q1 = np.quantile(volumes, 0.33)
    q2 = np.quantile(volumes, 0.66)

    if category == "small":
        return {code for code, volume in product_data if volume <= q1}
    elif category == "medium":
        return {code for code, volume in product_data if q1 < volume <= q2}
    else:  # large
        return {code for code, volume in product_data if volume > q2}


def get_customer_items_filtered(cust_id: str, allowed_product_codes: set) -> List[Item]:
    transaction_filepath = pathlib.Path() / "raw_json" / "Transaksi_v2.json"
    med_spec_filepath = pathlib.Path() / "raw_json" / "Dimensi_Suhu_v2.json"

    with open(transaction_filepath.absolute(), "r") as f:
        trans_dict = json.load(f)

    with open(med_spec_filepath.absolute(), "r") as f:
        med_spec_dict = json.load(f)

    items: List[Item] = []

    try:
        historical_transaction_dict = trans_dict[cust_id]
        r_date = random.choice(list(historical_transaction_dict.keys()))
        raw_item_dict_list = historical_transaction_dict[r_date]

        if not all(item["PRODUCT_CODE"] in allowed_product_codes for item in raw_item_dict_list):
            return []

        i = 0
        for raw_item_dict in raw_item_dict_list:
            product_code = raw_item_dict["PRODUCT_CODE"]
            qty = raw_item_dict["SHIPPED_QTY"]
            med_spec = med_spec_dict[product_code]
            weight = float(med_spec["berat_gram"])
            l, w, h = med_spec["panjang_cm"], med_spec["lebar_cm"], med_spec["tinggi_cm"]
            temp_req = med_spec["suhu_simpan"]
            dim = np.array([l, w, h], dtype=float)
            is_reefer_required = temp_req != "kamar"

            for _ in range(qty):
                items.append(Item(i, product_code, dim, weight, False, is_reefer_required))
                i += 1

    except KeyError:
        return []

    return items


def generate_customers(cabang: str, num_customers: int, size_category: str) -> List[Customer]:
    filename = cabang + ".json"
    filepath = pathlib.Path() / "raw_json" / filename
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)

    allowed_codes = get_product_codes_by_category(size_category)

    customers_dict_list = d["CUSTOMERS"]
    random.shuffle(customers_dict_list)

    customers: List[Customer] = []
    i = 0
    for customer_dict in customers_dict_list:
        if len(customers) >= num_customers:
            break
        c_coord = np.array([float(x) for x in customer_dict["Maps"].split(",")], dtype=float)
        cust_id = customer_dict["CUSTOMER_NUMBER"]
        items = get_customer_items_filtered(str(cust_id), allowed_codes)
        if items:
            customers.append(Customer(i + 1, cust_id, c_coord, items))
            i += 1

    return customers


def generate_vehicles(num_normal_trucks, num_reefer_trucks) -> List[Vehicle]:
    vehicle_filepath = pathlib.Path() / "raw_json" / "Kendaraan.json"
    with open(vehicle_filepath.absolute(), "r") as f:
        vec_dict = json.load(f)

    vehicles: List[Vehicle] = []
    i = 0

    for _ in range(num_normal_trucks):
        v = vec_dict["NORMAL_TRUCK"]
        dim = np.array([v["PANJANG"], v["LEBAR"], v["TINGGI"]], dtype=float)
        vehicles.append(Vehicle(i, v["SERVICE"], v["KAPASITAS"], dim, False, v["RATE"], v["VARIABLE_RATE"]))
        i += 1

    for _ in range(num_reefer_trucks):
        v = vec_dict["REEFER_TRUCK"]
        dim = np.array([v["PANJANG"], v["LEBAR"], v["TINGGI"]], dtype=float)
        vehicles.append(Vehicle(i, v["SERVICE"], v["KAPASITAS"], dim, False, v["RATE"], v["VARIABLE_RATE"]))
        i += 1

    return vehicles


def generate(cabang: str, num_customers: int, size_category: str,
             num_normal_trucks: int = 10,
             num_reefer_trucks: int = 10):
    filename = cabang + ".json"
    filepath = pathlib.Path() / "raw_json" / filename
    with open(filepath.absolute(), "r") as f:
        d = json.load(f)

    depot_coord = np.array([float(x) for x in d["CABANG"]["Maps"].split(",")], dtype=float)

    customers = generate_customers(cabang, num_customers, size_category)
    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(f"{cabang}_{size_category}")


if __name__ == "__main__":
    # Example usage: choose "small", "medium", or "large"
    generate("JK2", 30, size_category="large")
