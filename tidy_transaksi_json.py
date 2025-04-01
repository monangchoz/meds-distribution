import json
import pathlib

if __name__ == "__main__":
    filepath = pathlib.Path()/"raw_json"/"Transaksi.json"
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)
    new_d = {}
    for row in d:
        cust_id = int(row["CUSTOMER_NUMBER"])
        date = row["DISPATCH_DATE"]
        if cust_id not in new_d.keys():
            new_d[cust_id] = {}
        if date not in new_d[cust_id].keys():
            new_d[cust_id][date] = []
        new_d[cust_id][date].append({"PRODUCT_CODE": row["PRODUCT_CODE"], "SHIPPED_QTY":row["SHIPPED_QTY"]})        
    
    filepath = pathlib.Path()/"raw_json"/"Transaksi_v2.json"
    with open(filepath.absolute(), "w") as f:
        json.dump(new_d, f)