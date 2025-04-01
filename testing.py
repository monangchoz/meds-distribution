# Generate instance based on a certain randomly picked dates
import json
import pathlib
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import folium
import webbrowser


class Node:
    def __init__(self, idx: int, coord: np.ndarray):
        self.idx = idx
        self.coord = coord

class Customer(Node):
    def __init__(self, idx, coord, cust_id: int, products: List[Dict] = None):
        super().__init__(idx, coord)
        self.cust_id = cust_id
        self.products = products if products is not None else []

    def get_products_dict(self) -> Dict[str, int]:
        return {p["PRODUCT_CODE"]: p["SHIPPED_QTY"] for p in self.products}


def generate(cabang: str, num_customers: int):
    filename = cabang + ".json"
    filepath = pathlib.Path() / "raw_json" / filename
    transaksi_path = pathlib.Path("raw_json") / "Transaksi.json"

    depot: Node
    customers: List[Customer] = []

    # Load depot + customer location data
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)

    depot_coord = d["CABANG"]["Maps"].split(",")
    depot_coord = np.asanyarray([float(c) for c in depot_coord], dtype=float)
    depot = Node(0, depot_coord)

    customers_dict_list = d["CUSTOMERS"]

    # Load transaction data
    with open(transaksi_path.absolute(), "r") as trans_data:
        transaksi_list = json.load(trans_data)

    # Step 1: Get all unique dispatch dates
    dates = list(set(entry["DISPATCH_DATE"] for entry in transaksi_list))

    chosen_date = None
    filtered_trans = []
    cust_products: Dict[int, List[Dict]] = {}

    # Step 2: Loop until we find a valid transaction date
    while True:
        chosen_date = np.random.choice(dates)
        print(f"📅 Trying date: {chosen_date}")

        filtered_trans = [entry for entry in transaksi_list if entry["DISPATCH_DATE"] == chosen_date]

        temp_cust_products: Dict[int, List[Dict]] = {}

        for entry in filtered_trans:
            cid = entry["CUSTOMER_NUMBER"]
            if cid not in temp_cust_products:
                temp_cust_products[cid] = []
            temp_cust_products[cid].append({
                "PRODUCT_CODE": entry["PRODUCT_CODE"],
                "SHIPPED_QTY": entry["SHIPPED_QTY"]
            })

        if temp_cust_products:
            cust_products = temp_cust_products
            break

    print(f"✅ Using transactions from date: {chosen_date}")

    # Filter customer_dicts based on selected transactions
    valid_customer_numbers = set(cust_products.keys())
    filtered_customer_dicts = [c for c in customers_dict_list if c["CUSTOMER_NUMBER"] in valid_customer_numbers]

    if len(filtered_customer_dicts) < num_customers:
        print(f"⚠️ Only {len(filtered_customer_dicts)} customers found on {chosen_date}. Reducing sample size.")
        num_customers = len(filtered_customer_dicts)

    chosen_customers = np.random.choice(filtered_customer_dicts, size=num_customers, replace=False)

    # Create Customer objects
    for i, customer_dict in enumerate(chosen_customers):
        c_coord = customer_dict["Maps"].split(",")
        c_coord = np.asanyarray([float(c) for c in c_coord], dtype=float)
        cust_id = customer_dict["CUSTOMER_NUMBER"]

        products = cust_products.get(cust_id, [])
        new_cust = Customer(i + 1, c_coord, cust_id, products)
        customers.append(new_cust)

        print(f"Customer {cust_id} has:")
        for code, qty in new_cust.get_products_dict().items():
            print(f"  - {code}: {qty}")

    # Build the map
    map = folium.Map((depot.coord[0], depot.coord[1]), zoom_start=12)

    folium.Marker(
        location=[depot.coord[0], depot.coord[1]],
        popup="Depot",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(map)

    for cust in customers:
        product_info = "<br>".join([f'{p["PRODUCT_CODE"]}: {p["SHIPPED_QTY"]}' for p in cust.products]) or "No products"
        popup_text = f"Customer ID: {cust.cust_id}<br>{product_info}"
        folium.Marker(
            location=[cust.coord[0], cust.coord[1]],
            popup=popup_text
        ).add_to(map)

    map.save("map.html")
    webbrowser.open("map.html")

    '''
    plt.scatter(cust_coords[:, 0], cust_coords[:, 1])
    plt.scatter(depot_coord[0], depot_coord[1])
    plt.show()
    '''

if __name__ == "__main__":
    generate("JK2", 200)
