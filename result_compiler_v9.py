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

def create_mks_running_time_analysis(df):
    """
    Running Time Analysis for MKS Zone Only (Historical Data, 15 Customers, 1 Clusters)
    """
    # Filter data for MKS zone only
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 15) & 
        (df['num_clusters'] == 1) &
        (df['location'] == 'MKS')
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for MKS zone with historical, 15 customers (rounded), 1 clusters")
        return None
    
    print(f"📊 MKS Zone Analysis - Found {len(filtered_df)} records:")
    print(f"  - Zone: MKS")
    print(f"  - Data type: historical")
    print(f"  - Customers: 15 (rounded)")
    print(f"  - Original customer counts: {sorted(filtered_df['num_customers'].unique())}")
    print(f"  - Clusters: 1")
    print(f"  - Algorithms: {sorted(filtered_df['algorithm'].unique())}")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group data by algorithm, calculate statistics
    mks_analysis = filtered_df.groupby('algorithm').agg({
        'running_time': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    mks_analysis.columns = ['algorithm', 'time_mean', 'time_std', 'count']
    
    # Sort by mean running time for better visualization
    mks_analysis = mks_analysis.sort_values('time_mean')
    
    # Set up bar chart
    algorithms = mks_analysis['algorithm'].tolist()
    times = mks_analysis['time_mean'].tolist()
    stds = mks_analysis['time_std'].tolist()
    
    # Create bars without error bars
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(range(len(algorithms)), times, 
                  color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_title('Running Time Analysis - MKS Zone\n(Historical Data, 15 Customers, 1 Cluster)', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel('Running Time (seconds)', fontsize=14)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical information
    total_records = len(filtered_df)
    avg_records_per_alg = filtered_df.groupby('algorithm').size().mean()
    
    plt.tight_layout()
    return fig

def create_sby_running_time_analysis(df):
    """
    Running Time Analysis for SBY Zone Only (Historical Data, 15 Customers, 1 Clusters)
    """
    # Filter data for SBY zone only
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 15) & 
        (df['num_clusters'] == 1) &
        (df['location'] == 'SBY')
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for SBY zone with historical, 15 customers (rounded), 1 clusters")
        return None
    
    print(f"📊 SBY Zone Analysis - Found {len(filtered_df)} records:")
    print(f"  - Zone: SBY")
    print(f"  - Data type: historical")
    print(f"  - Customers: 15 (rounded)")
    print(f"  - Original customer counts: {sorted(filtered_df['num_customers'].unique())}")
    print(f"  - Clusters: 1")
    print(f"  - Algorithms: {sorted(filtered_df['algorithm'].unique())}")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group data by algorithm, calculate statistics
    sby_analysis = filtered_df.groupby('algorithm').agg({
        'running_time': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    sby_analysis.columns = ['algorithm', 'time_mean', 'time_std', 'count']
    
    # Sort by mean running time for better visualization
    sby_analysis = sby_analysis.sort_values('time_mean')
    
    # Set up bar chart
    algorithms = sby_analysis['algorithm'].tolist()
    times = sby_analysis['time_mean'].tolist()
    stds = sby_analysis['time_std'].tolist()
    
    # Create bars without error bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(range(len(algorithms)), times, 
                  color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_title('Running Time Analysis - SBY Zone\n(Historical Data, 15 Customers, 1 Cluster)', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel('Running Time (seconds)', fontsize=14)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical information
    total_records = len(filtered_df)
    avg_records_per_alg = filtered_df.groupby('algorithm').size().mean()
    
    plt.tight_layout()
    return fig

def create_mks_sby_comparison_analysis(df):
    """
    Side-by-side comparison of MKS vs SBY zones
    """
    # Filter data for both MKS and SBY
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 15) & 
        (df['num_clusters'] == 1) &
        (df['location'].isin(['MKS', 'SBY']))
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for MKS and SBY zones")
        return None
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # MKS Analysis
    mks_data = filtered_df[filtered_df['location'] == 'MKS']
    if not mks_data.empty:
        mks_analysis = mks_data.groupby('algorithm').agg({
            'running_time': ['mean', 'std']
        }).reset_index()
        mks_analysis.columns = ['algorithm', 'time_mean', 'time_std']
        mks_analysis = mks_analysis.sort_values('time_mean')
        
        algorithms_mks = mks_analysis['algorithm'].tolist()
        times_mks = mks_analysis['time_mean'].tolist()
        stds_mks = mks_analysis['time_std'].tolist()
        
        colors_mks = plt.cm.Set2(np.linspace(0, 1, len(algorithms_mks)))
        bars_mks = ax1.bar(range(len(algorithms_mks)), times_mks,
                          color=colors_mks, alpha=0.8, 
                          edgecolor='black', linewidth=1)
        
        ax1.set_title('MKS Zone\n(Historical, 15 Customers, 1 Cluster)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Running Time (seconds)', fontsize=12)
        ax1.set_xticks(range(len(algorithms_mks)))
        ax1.set_xticklabels(algorithms_mks, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # SBY Analysis
    sby_data = filtered_df[filtered_df['location'] == 'SBY']
    if not sby_data.empty:
        sby_analysis = sby_data.groupby('algorithm').agg({
            'running_time': ['mean', 'std']
        }).reset_index()
        sby_analysis.columns = ['algorithm', 'time_mean', 'time_std']
        sby_analysis = sby_analysis.sort_values('time_mean')
        
        algorithms_sby = sby_analysis['algorithm'].tolist()
        times_sby = sby_analysis['time_mean'].tolist()
        stds_sby = sby_analysis['time_std'].tolist()
        
        colors_sby = plt.cm.Set3(np.linspace(0, 1, len(algorithms_sby)))
        bars_sby = ax2.bar(range(len(algorithms_sby)), times_sby,
                          color=colors_sby, alpha=0.8, 
                          edgecolor='black', linewidth=1)
        
        ax2.set_title('SBY Zone\n(Historical, 15 Customers, 1 Cluster)', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Running Time (seconds)', fontsize=12)
        ax2.set_xticks(range(len(algorithms_sby)))
        ax2.set_xticklabels(algorithms_sby, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Running Time Comparison: MKS vs SBY Zones', 
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig

def print_zone_specific_insights(df):
    """
    Print detailed insights for MKS and SBY zones
    """
    # Filter data for both zones
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 15) & 
        (df['num_clusters'] == 1) &
        (df['location'].isin(['MKS', 'SBY']))
    ].copy()
    
    if filtered_df.empty:
        print("❌ No data available for zone-specific insights")
        return
    
    print("\n" + "=" * 70)
    print("📊 ZONE-SPECIFIC INSIGHTS (MKS vs SBY)")
    print("=" * 70)
    
    for zone in ['MKS', 'SBY']:
        zone_data = filtered_df[filtered_df['location'] == zone]
        if zone_data.empty:
            continue
            
        print(f"\n🏙️ {zone} Zone Analysis:")
        print("-" * 25)
        
        # Check if there are algorithms
        alg_means = zone_data.groupby('algorithm')['running_time'].mean()
        if alg_means.empty:
            print(f"   No algorithm data available for {zone}")
            continue
        
        # Best algorithm
        best_alg = alg_means.idxmin()
        best_time = alg_means.min()
        print(f"🏆 Best algorithm: {best_alg} ({best_time:.3f}s)")
        
        # Worst algorithm
        worst_alg = alg_means.idxmax()
        worst_time = alg_means.max()
        print(f"🐌 Slowest algorithm: {worst_alg} ({worst_time:.3f}s)")
        
        # Algorithm ranking
        alg_ranking = alg_means.sort_values()
        print(f"📈 Algorithm ranking:")
        for i, (alg, time) in enumerate(alg_ranking.items(), 1):
            print(f"   {i}. {alg}: {time:.3f}s")
        
        # Statistical summary
        print(f"📊 Statistics:")
        print(f"   Average time: {zone_data['running_time'].mean():.3f}s")
        print(f"   Standard deviation: {zone_data['running_time'].std():.3f}s")
        print(f"   Min time: {zone_data['running_time'].min():.3f}s")
        print(f"   Max time: {zone_data['running_time'].max():.3f}s")
    
    # Cross-zone comparison
    zones_with_data = filtered_df['location'].unique()
    if len(zones_with_data) >= 2:
        print(f"\n🔄 Cross-Zone Comparison:")
        print("-" * 25)
        zone_averages = filtered_df.groupby('location')['running_time'].mean()
        faster_zone = zone_averages.idxmin()
        slower_zone = zone_averages.idxmax()
        
        print(f"⚡ Faster zone: {faster_zone} ({zone_averages[faster_zone]:.3f}s avg)")
        print(f"🐢 Slower zone: {slower_zone} ({zone_averages[slower_zone]:.3f}s avg)")
        
        if len(zone_averages) == 2:
            speed_diff = zone_averages[slower_zone] - zone_averages[faster_zone]
            speed_pct = (speed_diff / zone_averages[faster_zone]) * 100
            print(f"📊 Difference: {speed_diff:.3f}s ({speed_pct:.1f}% slower)")

def main():
    """Main function for zone-specific analysis"""
    base_path = os.getcwd()
    
    print("🔍 Zone-Specific Analysis: MKS and SBY")
    print("=" * 50)
    print("Filtering criteria:")
    print("  - Data type: Historical")
    print("  - Number of customers: 15 (rounded)")
    print("  - Number of clusters: 1")
    print("  - Zones: MKS and SBY only")
    print("=" * 50)
    
    # Load data
    df = load_zone_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure.")
        print("Expected structure:")
        print("  current_directory/")
        print("  ├── results/")
        print("  │   ├── ga/")
        print("  │   ├── de/")
        print("  │   ├── brkga/")
        print("  │   └── pso/")
        return
    
    print(f"\n📊 Total records loaded: {len(df)}")
    
    # Check available zones
    available_zones = df['location'].unique()
    print(f"📍 Available zones: {sorted([z for z in available_zones if z is not None])}")
    
    # Check if MKS or SBY data exists
    has_mks = 'MKS' in available_zones
    has_sby = 'SBY' in available_zones
    
    if not has_mks and not has_sby:
        print("❌ No MKS or SBY data found in the dataset")
        return
    
    # Create MKS-specific analysis
    if has_mks:
        print("\n📈 Creating MKS zone analysis...")
        fig_mks = create_mks_running_time_analysis(df)
        if fig_mks:
            plt.show()
    else:
        print("\n⚠️ No MKS data available")
        fig_mks = None
    
    # Create SBY-specific analysis  
    if has_sby:
        print("\n📈 Creating SBY zone analysis...")
        fig_sby = create_sby_running_time_analysis(df)
        if fig_sby:
            plt.show()
    else:
        print("\n⚠️ No SBY data available")
        fig_sby = None
    
    # Create comparison analysis
    if has_mks and has_sby:
        print("\n📈 Creating MKS vs SBY comparison...")
        fig_comparison = create_mks_sby_comparison_analysis(df)
        if fig_comparison:
            plt.show()
    else:
        print("\n⚠️ Cannot create comparison - need both MKS and SBY data")
        fig_comparison = None
    
    # Print insights
    print_zone_specific_insights(df)
    
    # Save option
    if fig_mks or fig_sby or fig_comparison:
        save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
        
        if save_plots == 'y':
            if fig_mks:
                fig_mks.savefig('mks_zone_analysis.png', dpi=300, bbox_inches='tight')
                print("✅ Saved mks_zone_analysis.png")
            
            if fig_sby:
                fig_sby.savefig('sby_zone_analysis.png', dpi=300, bbox_inches='tight')
                print("✅ Saved sby_zone_analysis.png")
            
            if fig_comparison:
                fig_comparison.savefig('mks_sby_comparison.png', dpi=300, bbox_inches='tight')
                print("✅ Saved mks_sby_comparison.png")
            
            print("\n🎉 Zone-specific analysis complete!")

if __name__ == "__main__":
    main()