import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Read the Excel file
file_path = 'TGR1.xlsx'  # Replace with your actual file path
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Robust coordinate parsing function with error handling
def parse_coordinates(coord_str):
    try:
        # Handle cases where coordinates might have brackets, spaces, etc.
        cleaned = str(coord_str).strip('()[]{} \t\n').replace(' ', '')
        lat, lon = cleaned.split(',')
        return (float(lat), float(lon))
    except Exception as e:
        print(f"Error parsing coordinates: {coord_str} - {str(e)}")
        return (np.nan, np.nan)

# Apply the parsing function to both coordinate columns
df['customer_coords'] = df['Maps'].apply(parse_coordinates)
df['depot_coords'] = df['Maps_2'].apply(parse_coordinates)

# Check for parsing errors
if df['customer_coords'].isna().any() or df['depot_coords'].isna().any():
    print("\nWarning: Some coordinates couldn't be parsed. Rows with NaN coordinates will be dropped.")
    print(f"Rows before dropping: {len(df)}")
    df = df.dropna(subset=['customer_coords', 'depot_coords'])
    print(f"Rows after dropping: {len(df)}")

# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    try:
        return geodesic(coord1, coord2).km
    except Exception as e:
        print(f"Distance calculation error: {str(e)}")
        return np.nan

# Calculate distances from depot
df['distance_from_depot'] = df.apply(
    lambda row: calculate_distance(row['customer_coords'], row['depot_coords']), 
    axis=1
)

# Calculate IQR for distances
Q1 = df['distance_from_depot'].quantile(0.25)
Q3 = df['distance_from_depot'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds (1.5 is typical, adjust as needed)
lower_bound = max(0, Q1 - 1.5 * IQR)  # Distance can't be negative
upper_bound = Q3 + 0.5 * IQR

# Filter outliers
df_clean = df[(df['distance_from_depot'] >= lower_bound) & 
              (df['distance_from_depot'] <= upper_bound)]

# Visualization
plt.figure(figsize=(12, 8))

# Plot all customer points
plt.scatter(
    [coord[1] for coord in df['customer_coords']],  # longitude
    [coord[0] for coord in df['customer_coords']],  # latitude
    c='blue', label='All Customers', alpha=0.6
)

# Plot depot location (using first depot coordinate)
depot_coord = df['depot_coords'].iloc[0]
plt.scatter(
    depot_coord[1], depot_coord[0], 
    c='red', marker='*', s=300, label='Depot', edgecolor='black'
)

# Plot outliers
outliers = df[(df['distance_from_depot'] < lower_bound) | 
              (df['distance_from_depot'] > upper_bound)]
plt.scatter(
    [coord[1] for coord in outliers['customer_coords']],
    [coord[0] for coord in outliers['customer_coords']],
    c='orange', marker='x', s=100, linewidths=2, label='Outliers'
)

# Add distance rings around depot for reference
for distance in [lower_bound, upper_bound]:
    circle = plt.Circle(
        (depot_coord[1], depot_coord[0]), 
        distance/111,  # Approximate conversion from km to degrees
        color='gray', fill=False, linestyle='--', alpha=0.5
    )
    plt.gca().add_patch(circle)
    plt.text(
        depot_coord[1], depot_coord[0] + distance/111, 
        f'{distance:.1f} km', ha='center', va='bottom', color='gray'
    )

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Customer Locations with Outliers Highlighted\n(IQR Method)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Print summary statistics
print("\n" + "="*50)
print("Outlier Removal Summary")
print("="*50)
print(f"Original data points: {len(df)}")
print(f"Data points after outlier removal: {len(df_clean)}")
print(f"Number of outliers removed: {len(df) - len(df_clean)}")
print(f"\nDistance statistics (km):")
print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print("\nOutlier distances:")
print(outliers['distance_from_depot'].describe())

plt.show()

# Optionally save the cleaned data
df_clean.to_excel('TGR2.xlsx', index=False)