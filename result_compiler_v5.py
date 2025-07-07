import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
from matplotlib.ticker import FuncFormatter

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

def load_data(base_path):
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
                
                print(f"✓ Processed {filename} -> {algorithm_name}, {location}, {data_type}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    return df

def create_data_type_zone_comparison(df):
    """
    Create multipanel comparison of Generated vs Historical data running time by zone
    """
    if df.empty:
        print("No data available")
        return None
    
    # Filter out records with missing essential data
    df_clean = df.dropna(subset=['data_type', 'location', 'algorithm', 'running_time']).copy()
    
    if df_clean.empty:
        print("No clean data available")
        return None
    
    print(f"📊 Creating data type comparison:")
    print(f"  - Data types: {sorted(df_clean['data_type'].unique())}")
    print(f"  - Zones: {sorted(df_clean['location'].unique())}")
    print(f"  - Algorithms: {sorted(df_clean['algorithm'].unique())}")
    print(f"  - Total records: {len(df_clean)}")
    
    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get unique zones and algorithms
    zones = sorted(df_clean['location'].unique())
    algorithms = sorted(df_clean['algorithm'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # Bar chart parameters
    x = np.arange(len(zones))
    width = 0.2
    
    # === LEFT PANEL: GENERATED DATA ===
    generated_data = df_clean[df_clean['data_type'].str.lower() == 'generated']
    
    if not generated_data.empty:
        # Group by zone and algorithm, calculate mean running time
        generated_analysis = generated_data.groupby(['location', 'algorithm']).agg({
            'running_time': 'mean'
        }).reset_index()
        
        generated_analysis.columns = ['location', 'algorithm', 'time_mean']
        
        # Create bars for each algorithm
        for i, alg in enumerate(algorithms):
            alg_data = generated_analysis[generated_analysis['algorithm'] == alg]
            
            times = []
            for zone in zones:
                zone_data = alg_data[alg_data['location'] == zone]
                if not zone_data.empty:
                    times.append(zone_data['time_mean'].iloc[0])
                else:
                    times.append(0)
            
            offset = (i - len(algorithms)/2 + 0.5) * width
            ax1.bar(x + offset, times, width, label=alg, color=colors[i], alpha=0.8)
        
        ax1.set_title('Generated Data\nRunning Time by Zone', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Zone', fontsize=12)
        ax1.set_ylabel('Running Time (seconds)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(zones, fontsize=11)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        
        print(f"  ✓ Generated data: {len(generated_data)} records")
    else:
        ax1.text(0.5, 0.5, 'No Generated Data Available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14, style='italic')
        ax1.set_title('Generated Data\nRunning Time by Zone', fontweight='bold', fontsize=14)
    
    # === RIGHT PANEL: HISTORICAL DATA ===
    historical_data = df_clean[df_clean['data_type'].str.lower() == 'historical']
    
    if not historical_data.empty:
        # Group by zone and algorithm, calculate mean running time
        historical_analysis = historical_data.groupby(['location', 'algorithm']).agg({
            'running_time': 'mean'
        }).reset_index()
        
        historical_analysis.columns = ['location', 'algorithm', 'time_mean']
        
        # Create bars for each algorithm
        for i, alg in enumerate(algorithms):
            alg_data = historical_analysis[historical_analysis['algorithm'] == alg]
            
            times = []
            for zone in zones:
                zone_data = alg_data[alg_data['location'] == zone]
                if not zone_data.empty:
                    times.append(zone_data['time_mean'].iloc[0])
                else:
                    times.append(0)
            
            offset = (i - len(algorithms)/2 + 0.5) * width
            ax2.bar(x + offset, times, width, label=alg, color=colors[i], alpha=0.8)
        
        ax2.set_title('Historical Data\nRunning Time by Zone', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Zone', fontsize=12)
        ax2.set_ylabel('Running Time (seconds)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(zones, fontsize=11)
        ax2.tick_params(axis='y', labelsize=11)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
        
        print(f"  ✓ Historical data: {len(historical_data)} records")
    else:
        ax2.text(0.5, 0.5, 'No Historical Data Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, style='italic')
        ax2.set_title('Historical Data\nRunning Time by Zone', fontweight='bold', fontsize=14)
    
    # Main title
    plt.suptitle('Algorithm Performance Comparison: Generated vs Historical Data by Zone', 
                fontweight='bold', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def create_summary_statistics_table(df):
    """
    Create summary statistics table comparing generated vs historical data
    """
    if df.empty:
        return None
    
    df_clean = df.dropna(subset=['data_type', 'location', 'algorithm', 'running_time']).copy()
    
    if df_clean.empty:
        return None
    
    # Calculate statistics for both data types
    stats_summary = df_clean.groupby(['data_type', 'location', 'algorithm']).agg({
        'running_time': ['mean', 'std', 'count'],
        'total_cost': ['mean']
    }).round(3)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Data Type', 'Zone', 'Algorithm', 'Avg Time (s)', 'Std Time', 'Count', 'Avg Cost']
    
    data_types = ['generated', 'historical']
    for data_type in data_types:
        data_subset = df_clean[df_clean['data_type'].str.lower() == data_type]
        
        if not data_subset.empty:
            for zone in sorted(data_subset['location'].unique()):
                for alg in sorted(data_subset['algorithm'].unique()):
                    subset = data_subset[(data_subset['location'] == zone) & (data_subset['algorithm'] == alg)]
                    
                    if not subset.empty:
                        avg_time = subset['running_time'].mean()
                        std_time = subset['running_time'].std()
                        count = len(subset)
                        avg_cost = subset['total_cost'].mean()
                        
                        row = [
                            data_type.title(),
                            zone,
                            alg,
                            f"{avg_time:.3f}",
                            f"{std_time:.3f}" if not pd.isna(std_time) else "0.000",
                            str(count),
                            f"{avg_cost:,.0f}"
                        ]
                        table_data.append(row)
    
    if table_data:
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.08, 0.12, 0.12, 0.10, 0.08, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternate data types
        current_type = None
        color_toggle = False
        for i in range(1, len(table_data) + 1):
            if table_data[i-1][0] != current_type:  # New data type
                current_type = table_data[i-1][0]
                color_toggle = not color_toggle
            
            for j in range(len(headers)):
                if current_type == 'Generated':
                    table[(i, j)].set_facecolor('#e3f2fd' if color_toggle else '#ffffff')
                else:  # Historical
                    table[(i, j)].set_facecolor('#f3e5f5' if color_toggle else '#ffffff')
        
        plt.title('Performance Statistics: Generated vs Historical Data by Zone', 
                  fontsize=16, fontweight='bold', pad=30)
    
    return fig

def main():
    """Main function to run the data type comparison analysis"""
    base_path = os.getcwd()
    
    print("🔍 Generated vs Historical Data Comparison by Zone")
    print("=" * 70)
    print("Analyzing running time performance across:")
    print("  - Data types: Generated vs Historical")
    print("  - Zones: JK2, MKS, SBY")
    print("  - Algorithms: GA, DE, BRKGA, PSO")
    print("=" * 70)
    
    # Load data
    df = load_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure.")
        return
    
    print(f"\n📊 Total records loaded: {len(df)}")
    
    # Create visualizations
    print("\n📈 Creating data type comparison plot...")
    fig1 = create_data_type_zone_comparison(df)
    if fig1:
        plt.show()
    
    print("\n📋 Creating summary statistics table...")
    fig2 = create_summary_statistics_table(df)
    if fig2:
        plt.show()
    
    # Print insights
    df_clean = df.dropna(subset=['data_type', 'location', 'algorithm', 'running_time'])
    
    if not df_clean.empty:
        print("\n" + "=" * 70)
        print("📊 KEY INSIGHTS")
        print("=" * 70)
        
        # Compare data types overall
        data_type_avg = df_clean.groupby('data_type')['running_time'].mean().sort_values()
        print("⚡ Average running time by data type:")
        for data_type, avg_time in data_type_avg.items():
            print(f"  {data_type.title()}: {avg_time:.3f}s")
        
        # Best algorithm for each data type
        print(f"\n🏆 Best algorithm per data type:")
        for data_type in df_clean['data_type'].unique():
            subset = df_clean[df_clean['data_type'] == data_type]
            best_alg = subset.groupby('algorithm')['running_time'].mean().idxmin()
            best_time = subset.groupby('algorithm')['running_time'].mean().min()
            print(f"  {data_type.title()}: {best_alg} ({best_time:.3f}s)")
        
        # Zone performance by data type
        print(f"\n📍 Zone performance by data type:")
        for data_type in sorted(df_clean['data_type'].unique()):
            subset = df_clean[df_clean['data_type'] == data_type]
            if not subset.empty:
                zone_avg = subset.groupby('location')['running_time'].mean().sort_values()
                print(f"  {data_type.title()} data - Best to Worst zones:")
                for i, (zone, avg_time) in enumerate(zone_avg.items(), 1):
                    print(f"    {i}. {zone}: {avg_time:.3f}s")
    
    # Save option
    save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
    
    if save_plots == 'y':
        if fig1:
            fig1.savefig('data_type_zone_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Saved data_type_zone_comparison.png")
        
        if fig2:
            fig2.savefig('data_type_statistics_table.png', dpi=300, bbox_inches='tight')
            print("✅ Saved data_type_statistics_table.png")
        
        print("\n🎉 Data type comparison analysis complete!")

if __name__ == "__main__":
    main()