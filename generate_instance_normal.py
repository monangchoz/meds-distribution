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


def get_customer_items_if_normal_temp_only(cust_id: str) -> List[Item]:
    """Returns items only if all of the customer's items are room-temperature ('kamar')."""
    transaction_filepath = pathlib.Path() / "raw_json" / "Transaksi_v2.json"
    med_spec_filepath = pathlib.Path() / "raw_json" / "Dimensi_Suhu_v2.json"

    with open(transaction_filepath.absolute(), "r") as f:
        trans_dict = json.load(f)
    with open(med_spec_filepath.absolute(), "r") as f:
        med_spec_dict = json.load(f)

    items: List[Item] = []

    try:
        transaction_data = trans_dict[cust_id]
        r_date = random.choice(list(transaction_data.keys()))
        raw_items = transaction_data[r_date]

        # Only proceed if all products are room temperature
        if not all(med_spec_dict[item["PRODUCT_CODE"]]["suhu_simpan"] == "kamar" for item in raw_items):
            return []

        i = 0
        for raw_item in raw_items:
            product_code = raw_item["PRODUCT_CODE"]
            qty = raw_item["SHIPPED_QTY"]
            spec = med_spec_dict[product_code]
            weight = float(spec["berat_gram"])
            l, w, h = spec["panjang_cm"], spec["lebar_cm"], spec["tinggi_cm"]
            dim = np.array([l, w, h], dtype=float)
            temp_req = spec["suhu_simpan"]
            is_reefer_required = temp_req != "kamar"

            for _ in range(qty):
                items.append(Item(i, product_code, dim, weight, False, is_reefer_required))
                i += 1
    except:
        return []

    return items


def generate_customers_normal_only(cabang: str, num_customers: int) -> List[Customer]:
    filepath = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(filepath.absolute(), "r") as f:
        branch_data = json.load(f)

    all_customers = branch_data["CUSTOMERS"]
    random.shuffle(all_customers)

    customers: List[Customer] = []
    i = 0
    for cust in all_customers:
        if len(customers) >= num_customers:
            break
        coords = np.array([float(x) for x in cust["Maps"].split(",")])
        cust_id = cust["CUSTOMER_NUMBER"]
        items = get_customer_items_if_normal_temp_only(str(cust_id))
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


def generate_normal_temp_instance(cabang: str, num_customers: int,
                                   num_normal_trucks: int = 10,
                                   num_reefer_trucks: int = 5):
    filepath = pathlib.Path() / "raw_json" / f"{cabang}.json"
    with open(filepath.absolute(), "r") as f:
        branch_data = json.load(f)

    depot_coord = np.array([float(x) for x in branch_data["CABANG"]["Maps"].split(",")])

    customers = generate_customers_normal_only(cabang, num_customers)
    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)

    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(f"{cabang}_normal_only")


if __name__ == "__main__":
    generate_normal_temp_instance("JK2", 30)
