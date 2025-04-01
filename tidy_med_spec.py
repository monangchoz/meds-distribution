import json
import pathlib

if __name__ == "__main__":
    filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu.json"
    with open(filepath.absolute(), "r") as json_data:
        d = json.load(json_data)
    new_d = {}
    for row in d:
        product_code = (row["kode_produk"])
        new_d[product_code] = row
        
    filepath = pathlib.Path()/"raw_json"/"Dimensi_Suhu_v2.json"
    with open(filepath.absolute(), "w") as f:
        json.dump(new_d, f)