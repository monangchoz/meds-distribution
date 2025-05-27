import json

# Load the data
with open("TGR.json", "r", encoding="utf-8") as f:
    data = json.load(f)

customers = data["CUSTOMERS"]
print(f"Original customer count: {len(customers)}")

# Remove duplicates based on the 'Maps' coordinate
seen_coords = set()
unique_customers = []

for customer in customers:
    coord = customer.get("Maps")
    if coord and coord not in seen_coords:
        seen_coords.add(coord)
        unique_customers.append(customer)

# Update the data and save to new file
data["CUSTOMERS"] = unique_customers

with open("TGR_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Cleaned customer count: {len(unique_customers)}")
