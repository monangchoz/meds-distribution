import pandas as pd
import json  # Import the json module for pretty-printing

# Step 1: Read the Excel file
excel_file = "json_transaksi_per_pelanggan.xlsx"
df = pd.read_excel(excel_file)

# Step 2: Date column
date_columns = ["DISPATCH_DATE"]  

# Step 3: Convert date strings to proper datetime format and remove time
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%m/%d/%Y").dt.strftime("%m/%d/%Y")  # Format as date string

# Step 4: Dataframe > JSON
json_data = df.to_json(orient="records", date_format="iso")

# Convert JSON string to a Python object for pretty-printing
json_object = json.loads(json_data)

# Save JSON to a file with indentation
with open("output.json", "w") as f:
    json.dump(json_object, f, indent=4)  # Use indent=4 for pretty-printing

print(json.dumps(json_object, indent=4))  # Print the JSON data to verify