import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
from matplotlib.ticker import FuncFormatter

def round_customer_count(num_customers):
    """
    Round customer count to research values: 15, 30, or 50
    """
    if pd.isna(num_customers):
        return num_customers
    
    research_values = [15, 30, 50]
    # Find closest research value
    closest = min(research_values, key=lambda x: abs(x - num_customers))
    return closest

def extract_detailed_info(filename):
    """
    Extract detailed information from filename
    Example: JK2_nc_15_ncl_5_nv_30_generated_r_0.20_0.20_0.60
    Returns: location, num_customers, num_clusters, num_vehicles, data_type, ratios
    """
    filename_upper = filename.upper()
    
    # Extract location
    location = None
    if 'JK2' in filename_upper:
        location = 'JK2'
    elif 'MKS' in filename_upper:
        location = 'MKS'
    elif 'SBY' in filename_upper:
        location = 'SBY'
    
    # Extract number of customers
    nc_match = re.search(r'nc_(\d+)', filename, re.IGNORECASE)
    num_customers = int(nc_match.group(1)) if nc_match else None
    
    # Extract number of clusters
    ncl_match = re.search(r'ncl_(\d+)', filename, re.IGNORECASE)
    num_clusters = int(ncl_match.group(1)) if ncl_match else None
    
    # Extract number of vehicles
    nv_match = re.search(r'nv_(\d+)', filename, re.IGNORECASE)
    num_vehicles = int(nv_match.group(1)) if nv_match else None
    
    # Extract data type
    data_type = None
    if 'HISTORICAL' in filename_upper:
        data_type = 'historical'
    elif 'GENERATED' in filename_upper:
        data_type = 'generated'
    
    # Extract ratios (small, medium, large goods)
    ratio_match = re.search(r'r_([\d.]+)_([\d.]+)_([\d.]+)', filename, re.IGNORECASE)
    ratios = None
    if ratio_match:
        ratios = (float(ratio_match.group(1)), 
                 float(ratio_match.group(2)), 
                 float(ratio_match.group(3)))
    
    return location, num_customers, num_clusters, num_vehicles, data_type, ratios

def load_zone_data(base_path):
    """
    Load CSV files and organize by multiple enhanced dimensions
    """
    enhanced_data = []
    
    # Algorithm folders
    algorithm_folders = {
        'ga': 'GA', 
        'de': 'DE',
        'brkga': 'BRKGA',
        'pso': 'PSO'
    }
    
    results_path = os.path.join(base_path, 'results')
    
    if not os.path.exists(results_path):
        print(f"Results folder not found at: {results_path}")
        return pd.DataFrame()
    
    # Process each algorithm folder
    for folder_name, algorithm_name in algorithm_folders.items():
        algorithm_path = os.path.join(results_path, folder_name)
        
        if not os.path.exists(algorithm_path):
            print(f"Algorithm folder not found: {algorithm_path}")
            continue
            
        # Get all CSV files
        csv_files = glob.glob(os.path.join(algorithm_path, "*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Extract information from filename
            location, num_customers, num_clusters, num_vehicles, data_type, ratios = extract_detailed_info(filename)
            
            try:
                # Read CSV data
                df = pd.read_csv(file_path, header=None, names=['total_cost', 'running_time'])
                
                # Add metadata to each row
                for _, row in df.iterrows():
                    record = {
                        'algorithm': algorithm_name,
                        'location': location,
                        'num_customers': num_customers,
                        'num_customers_rounded': round_customer_count(num_customers),  # Add rounded version
                        'num_clusters': num_clusters,
                        'num_vehicles': num_vehicles,
                        'data_type': data_type,
                        'ratio_small': ratios[0] if ratios else None,
                        'ratio_medium': ratios[1] if ratios else None,
                        'ratio_large': ratios[2] if ratios else None,
                        'total_cost': row['total_cost'],
                        'running_time': row['running_time'],
                        'filename': filename
                    }
                    enhanced_data.append(record)
                
                print(f"✓ Processed {filename} -> {algorithm_name}, {location}, nc:{num_customers}→{round_customer_count(num_customers)}, ncl:{num_clusters}, {data_type}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    
    if not df.empty:
        print(f"\n📊 Customer count mapping applied:")
        customer_mapping = df.groupby(['num_customers', 'num_customers_rounded']).size().reset_index()
        customer_mapping = customer_mapping[['num_customers', 'num_customers_rounded']].drop_duplicates().sort_values('num_customers')
        for _, row in customer_mapping.iterrows():
            if not pd.isna(row['num_customers']) and not pd.isna(row['num_customers_rounded']):
                print(f"  {int(row['num_customers'])} → {int(row['num_customers_rounded'])}")
    
    return df

def create_zone_running_time_analysis(df):
    """
    Running Time Analysis by Zone for Historical Data, 50 Customers, 5 Clusters (using rounded customer count)
    """
    # Filter data based on requirements using rounded customer count
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 15) & 
        (df['num_clusters'] == 1)
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for historical, 50 customers (rounded), 5 clusters")
        return None
    
    print(f"📊 Found {len(filtered_df)} records matching criteria:")
    print(f"  - Data type: historical")
    print(f"  - Customers: 50 (rounded)")
    print(f"  - Original customer counts: {sorted(filtered_df['num_customers'].unique())}")
    print(f"  - Clusters: 5")
    print(f"  - Zones: {sorted(filtered_df['location'].unique())}")
    print(f"  - Algorithms: {sorted(filtered_df['algorithm'].unique())}")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data by zone and algorithm, calculate mean running time
    zone_analysis = filtered_df.groupby(['location', 'algorithm']).agg({
        'running_time': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    zone_analysis.columns = ['location', 'algorithm', 'time_mean', 'time_std', 'count']
    
    # Get unique zones and algorithms
    zones = sorted(zone_analysis['location'].unique())
    algorithms = sorted(zone_analysis['algorithm'].unique())
    
    # Set up bar chart
    x = np.arange(len(zones))
    width = 0.2
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # Create bars for each algorithm
    for i, alg in enumerate(algorithms):
        alg_data = zone_analysis[zone_analysis['algorithm'] == alg]
        
        times = []
        for zone in zones:
            zone_data = alg_data[alg_data['location'] == zone]
            if not zone_data.empty:
                times.append(zone_data['time_mean'].iloc[0])
            else:
                times.append(0)
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=alg, color=colors[i], alpha=0.8)
    
    # Customize the plot
    ax.set_title('Running Time Analysis by Zone\n(Historical Data, 15 Customers, 1 Clusters)', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Zone', fontsize=14)
    ax.set_ylabel('Running Time (seconds)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add some statistical information
    plt.figtext(0.02, 0.02, 
                f'Data points per algorithm-zone: {filtered_df.groupby(["algorithm", "location"]).size().mean():.1f} avg',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def create_zone_performance_table(df):
    """
    Create performance summary table for zone analysis (using rounded customer count)
    """
    # Filter data using rounded customer count
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 50) & 
        (df['num_clusters'] == 5)
    ].copy()
    
    if filtered_df.empty:
        return None
    
    # Calculate statistics
    stats = filtered_df.groupby(['location', 'algorithm']).agg({
        'running_time': ['mean', 'std', 'min', 'max', 'count'],
        'total_cost': ['mean', 'std']
    }).round(3)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Zone', 'Algorithm', 'Avg Time (s)', 'Std Time', 'Min Time', 'Max Time', 'Count', 'Avg Cost']
    
    for zone in sorted(filtered_df['location'].unique()):
        for alg in sorted(filtered_df['algorithm'].unique()):
            zone_alg_data = filtered_df[(filtered_df['location'] == zone) & (filtered_df['algorithm'] == alg)]
            if not zone_alg_data.empty:
                avg_time = zone_alg_data['running_time'].mean()
                std_time = zone_alg_data['running_time'].std()
                min_time = zone_alg_data['running_time'].min()
                max_time = zone_alg_data['running_time'].max()
                count = len(zone_alg_data)
                avg_cost = zone_alg_data['total_cost'].mean()
                
                row = [
                    zone,
                    alg,
                    f"{avg_time:.3f}",
                    f"{std_time:.3f}" if not pd.isna(std_time) else "0.000",
                    f"{min_time:.3f}",
                    f"{max_time:.3f}",
                    str(count),
                    f"{avg_cost:,.0f}"
                ]
                table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows and highlight zones
    current_zone = None
    color_toggle = False
    for i in range(1, len(table_data) + 1):
        if table_data[i-1][0] != current_zone:  # New zone
            current_zone = table_data[i-1][0]
            color_toggle = not color_toggle
        
        for j in range(len(headers)):
            if color_toggle:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Detailed Performance Statistics by Zone\n(Historical Data, 50 Customers - Rounded, 5 Clusters)', 
              fontsize=16, fontweight='bold', pad=30)
    return fig

def main():
    """Main function to run the zone-specific analysis"""
    base_path = os.getcwd()
    
    print("🔍 Zone-Specific Running Time Analysis")
    print("=" * 65)
    print("Filtering criteria:")
    print("  - Data type: Historical")
    print("  - Number of customers: 50 (includes rounded)")
    print("  - Customer rounding: 48→50, 24→30, 18→15, 12→15")
    print("  - Number of clusters: 5")
    print("=" * 65)
    
    # Load data
    df = load_zone_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure.")
        return
    
    print(f"\n📊 Total records loaded: {len(df)}")
    
    # Create visualizations
    print("\n📈 Creating zone running time analysis...")
    fig1 = create_zone_running_time_analysis(df)
    if fig1:
        plt.show()
    
    print("\n📋 Creating performance summary table...")
    fig2 = create_zone_performance_table(df)
    if fig2:
        plt.show()
    
    # Print insights using rounded customer count
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 50) & 
        (df['num_clusters'] == 5)
    ]
    
    if not filtered_df.empty:
        print("\n" + "=" * 65)
        print("📊 KEY INSIGHTS (Using Rounded Customer Counts)")
        print("=" * 65)
        
        # Show customer count mapping
        customer_info = filtered_df.groupby(['num_customers', 'num_customers_rounded']).size().reset_index()
        customer_info = customer_info[['num_customers', 'num_customers_rounded']].drop_duplicates().sort_values('num_customers')
        print("📋 Customer count mapping used:")
        for _, row in customer_info.iterrows():
            if not pd.isna(row['num_customers']):
                print(f"  {int(row['num_customers'])} → {int(row['num_customers_rounded'])}")
        print("")
        
        # Best performing algorithm per zone
        zone_best = filtered_df.groupby(['location', 'algorithm'])['running_time'].mean().groupby('location').idxmin()
        print("🏆 Fastest algorithm per zone:")
        for location, (_, algorithm) in zone_best.items():
            avg_time = filtered_df[(filtered_df['location'] == location) & 
                                 (filtered_df['algorithm'] == algorithm)]['running_time'].mean()
            print(f"  {location}: {algorithm} ({avg_time:.3f}s)")
        
        # Overall zone performance
        zone_avg = filtered_df.groupby('location')['running_time'].mean().sort_values()
        print(f"\n📍 Zone ranking (by average running time):")
        for i, (zone, avg_time) in enumerate(zone_avg.items(), 1):
            print(f"  {i}. {zone}: {avg_time:.3f}s")
        
        # Algorithm performance across all zones
        alg_avg = filtered_df.groupby('algorithm')['running_time'].mean().sort_values()
        print(f"\n🤖 Algorithm ranking (across all zones):")
        for i, (alg, avg_time) in enumerate(alg_avg.items(), 1):
            print(f"  {i}. {alg}: {avg_time:.3f}s")
    
    # Save option
    save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
    
    if save_plots == 'y':
        if fig1:
            fig1.savefig('zone_running_time_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved zone_running_time_analysis.png")
        
        if fig2:
            fig2.savefig('zone_performance_table.png', dpi=300, bbox_inches='tight')
            print("✅ Saved zone_performance_table.png")
        
        print("\n🎉 Zone analysis complete!")

if __name__ == "__main__":
    main()