import pandas as pd

# Step 1: Read the Excel file
excel_file = "TGR2.xlsx"  # Excel file path
sheet_name = "Sheet1"  # Sheet name

# Excel file > DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Step 2: Dataframe > JSON
json_data = df.to_json(orient="records", indent=4)

# Step 3: Save the JSON data to a file
json_file = "TGR3.json"  # JSON File name
with open(json_file, "w") as file:
    file.write(json_data)

print(f"Excel file '{excel_file}' has been converted to JSON and saved as '{json_file}'!")