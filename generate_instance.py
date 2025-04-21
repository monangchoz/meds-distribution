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
        for raw_item_dict in raw_item_dict_list:
            product_code = raw_item_dict["PRODUCT_CODE"]
            qty = raw_item_dict["SHIPPED_QTY"]
            med_spec = med_spec_dict[product_code]
            weight = float(med_spec["berat_gram"])
            l,w,h = med_spec["panjang_cm"], med_spec["lebar_cm"], med_spec["tinggi_cm"]
            temp_req = med_spec["suhu_simpan"]
            dim = np.asanyarray([l,w,h], dtype=float)
            is_reefer_required = temp_req != "kamar"
            for _ in range(qty):
                new_item = Item(i, product_code, dim, weight, False, is_reefer_required)
                items.append(new_item)
                i += 1
    except KeyError:
        # generate items randomly
        threshold = 0.05*5000000 #gram
        total_weight = 0.
        i = 0
        while total_weight < threshold:
            qty = random.randint(1, 10)
            product_code = random.choice(list(med_spec_dict.keys()))
            med_spec = med_spec_dict[product_code]
            weight = float(med_spec["berat_gram"])
            l,w,h = med_spec["panjang_cm"], med_spec["lebar_cm"], med_spec["tinggi_cm"]
            temp_req = med_spec["suhu_simpan"]
            dim = np.asanyarray([l,w,h], dtype=float)
            is_reefer_required = temp_req != "kamar"
            for _ in range(qty):
                new_item = Item(i, product_code, dim, weight, False, is_reefer_required)
                items.append(new_item)
                total_weight += weight
                i += 1

    return items
    
def generate_customers(cabang, num_customers)->List[Customer]:
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
        items = get_customer_items_random_date(str(cust_id))
        new_cust = Customer(i+1, cust_id, c_coord, items)
        customers += [new_cust]
    return customers
     
def generate(cabang: str, 
             num_customers: int,
             num_normal_trucks:int=2,
             num_reefer_trucks:int=2):
    filename = cabang+".json"
    filepath = pathlib.Path()/"raw_json"/filename
    customers: List[Node] = []
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)
        
    depot_coord = d["CABANG"]["Maps"]
    depot_coord = depot_coord.split(",")
    depot_coord = np.asanyarray([float(c) for c in depot_coord], dtype=float)
    
    customers = generate_customers(cabang, num_customers)    
    vehicles = generate_vehicles(num_normal_trucks, num_reefer_trucks)
    problem = HVRP3L(depot_coord, customers, vehicles)
    problem.to_json(cabang)
    

def generate_vehicles(num_normal_trucks, num_reefer_trucks)->List[Vehicle]:
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
                          True,
                          reefer_truck_dict["RATE"],
                          reefer_truck_dict["VARIABLE_RATE"])
        vehicles.append(new_vec)
        i+=1
    return vehicles

if __name__=="__main__":
    generate("JK2", 50)