import pandas as pd
import requests

# Step 1: Read Excel File
excel_file = "master_cabang.xlsx"  # Excel file path
sheet_name = "Sheet1"  # Sheet name 

# Excel File > Dataframe
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Step 2: Function to get coordinates using Geoapify
def get_coordinates(location, api_key):
    base_url = "https://api.geoapify.com/v1/geocode/search"
    params = {
        "text": location,
        "apiKey": api_key,
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            lat = data["features"][0]["properties"]["lat"]
            lon = data["features"][0]["properties"]["lon"]
            return lat, lon
    return None, None  

# Step 3: Add columns for latitude and longitude in the DataFrame
df["Latitude"] = None
df["Longitude"] = None

# Step 4: Fetch coordinates for each location
api_key = "28ae6f8f78e8408cae079ab1dd742a60"  # Geoapify API key
for index, row in df.iterrows():
    location = row["alamat"]  # Replace with the column name 
    latitude, longitude = get_coordinates(location, api_key)
    df.at[index, "Latitude"] = latitude
    df.at[index, "Longitude"] = longitude
    print(f"Processed: {location} -> ({latitude}, {longitude})")

# Step 5: Save the updated DataFrame to a new Excel file
output_file = "locations_with_coordinates.xlsx"
df.to_excel(output_file, index=False)

print(f"Coordinates have been added and saved to '{output_file}'!")