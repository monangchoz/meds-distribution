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


def get_customer_items_if_reefer_only(cust_id: str) -> List[Item]:
    """Returns items for the customer only if all products require reefer."""
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
        raw_item_list = historical_transaction_dict[r_date]

        if not all(med_spec_dict[item["PRODUCT_CODE"]]["suhu_simpan"] != "kamar" for item in raw_item_list):
            return []

        i = 0
        for raw_item in raw_item_list:
            product_code = raw_item["PRODUCT_CODE"]
            qty = raw_item["SHIPPED_QTY"]
            spec = med_spec_dict[product_code]

            weight = float(spec["berat_gram"])
            l, w, h = spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]
            temp_req = spec["suhu_simpan"]
            dim = np.array([l, w, h], dtype=float)
            is_reefer_required = temp_req != "kamar"

            for _ in range(qty):
                items.append(Item(i, product_code, dim, weight, False, is_reefer_required))
                i += 1
    except:
        return []

    return items


def generate_customers_reefer_only(cabang: str, num_customers: int) -> List[Customer]:
    filepath = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(filepath.absolute(), "r") as f:
        branch_data = json.load(f)

    customers_raw = branch_data["CUSTOMERS"]
    random.shuffle(customers_raw)

    customers: List[Customer] = []
    i = 0
    for customer_data in customers_raw:
        if len(customers) >= num_customers:
            break
        coords = np.array([float(x) for x in customer_data["Maps"].split(",")])
        cust_id = customer_data["CUSTOMER_NUMBER"]
        items = get_customer_items_if_reefer_only(str(cust_id))
        if items:
            customers.append(Customer(i + 1, cust_id, coords, items))
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


def generate_reefer_only_instance(cabang: str, num_customers: int,
                                   num_normal_trucks: int = 5,
                                   num_reefer_trucks: int = 10):
    branch_path = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(branch_path.absolute(), "r") as f:
        branch_data = json.load(f)

    depot_coord = np.array([float(x) for x in branch_data["CABANG"]["Maps"].split(",")])

    customers = generate_customers_reefer_only(cabang, num_customers)
    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)

    problem.to_json(f"{cabang}_reefer_only")


if __name__ == "__main__":
    generate_reefer_only_instance("JK2", 30)
