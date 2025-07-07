import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

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

def load_enhanced_data(base_path):
    """
    Load CSV files and organize by multiple enhanced dimensions
    """
    # Enhanced data structure
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
                
                print(f"✓ Processed {filename} -> {algorithm_name}, {location}, nc:{num_customers}→{round_customer_count(num_customers)}, nv:{num_vehicles}, {data_type}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    
    if not df.empty:
        # Calculate efficiency metrics
        df['cost_per_customer'] = df['total_cost'] / df['num_customers_rounded']
        df['time_per_customer'] = df['running_time'] / df['num_customers_rounded']
        
        # Create proper cargo mix category based on actual ratios
        def categorize_cargo_mix(row):
            if pd.isna(row['ratio_small']):
                return 'Unknown'
            
            small, medium, large = row['ratio_small'], row['ratio_medium'], row['ratio_large']
            
            # Round to avoid floating point precision issues
            small_r = round(small, 2)
            medium_r = round(medium, 2)
            large_r = round(large, 2)
            
            # Match the actual ratio patterns you mentioned
            if (small_r, medium_r, large_r) == (0.2, 0.2, 0.6):
                return '(0.2, 0.2, 0.6)'
            elif (small_r, medium_r, large_r) == (0.2, 0.6, 0.2):
                return '(0.2, 0.6, 0.2)'
            elif (small_r, medium_r, large_r) == (0.6, 0.2, 0.2):
                return '(0.2, 0.2, 0.6)'
            elif abs(small_r - 0.33) < 0.02 and abs(medium_r - 0.33) < 0.02 and abs(large_r - 0.33) < 0.02:
                return '(1/3, 1/3, 1/3)'
            else:
                return f'Other ({small_r}-{medium_r}-{large_r})'
        
        df['cargo_mix'] = df.apply(categorize_cargo_mix, axis=1)
        
        print(f"\n📊 Loaded {len(df)} records from {df['algorithm'].nunique()} algorithms")
        print(f"📍 Locations: {sorted(df['location'].dropna().unique())}")
        print(f"👥 Customer counts (original): {sorted(df['num_customers'].dropna().unique())}")
        print(f"👥 Customer counts (rounded): {sorted(df['num_customers_rounded'].dropna().unique())}")
        print(f"🚛 Vehicle counts: {sorted(df['num_vehicles'].dropna().unique())}")
        print(f"📦 Cargo mixes: {sorted(df['cargo_mix'].unique())}")
    
    return df

def format_large_number(x, pos):
    """Format large numbers for better readability with more precision for close values"""
    if x >= 1000000:
        # More precision for millions to show small differences
        return f'{x/1000000:.2f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'

# ============= SCALABILITY ANALYSIS (4 separate plots) =============

def create_running_time_scalability(df):
    """Running Time vs Number of Customers (Average only, no variance)"""
    if df.empty or df['num_customers_rounded'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data using rounded customer counts and only get mean
    scalability_data = df.groupby(['algorithm', 'num_customers_rounded']).agg({
        'running_time': 'mean'
    }).reset_index()
    
    scalability_data.columns = ['algorithm', 'num_customers_rounded', 'time_mean']
    
    algorithms = scalability_data['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = scalability_data[scalability_data['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_customers_rounded'], alg_data['time_mean'], 
                   label=alg, marker='o', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Algorithm Scalability: Running Time vs Number of Customers', 
                fontweight='bold', fontsize=18)
    ax.set_xlabel('Number of Customers', fontsize=16)
    ax.set_ylabel('Running Time (seconds)', fontsize=16)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show only research values
    ax.set_xticks([15, 30, 50])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    return fig

def create_total_cost_scalability(df):
    """Total Cost vs Number of Customers (Average only, no variance)"""
    if df.empty or df['num_customers_rounded'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data using rounded customer counts and only get mean
    scalability_data = df.groupby(['algorithm', 'num_customers_rounded']).agg({
        'total_cost': 'mean'
    }).reset_index()
    
    scalability_data.columns = ['algorithm', 'num_customers_rounded', 'cost_mean']
    
    algorithms = scalability_data['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = scalability_data[scalability_data['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_customers_rounded'], alg_data['cost_mean'], 
                   label=alg, marker='s', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Algorithm Scalability: Total Cost vs Number of Customers', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Number of Customers', fontsize=14)
    ax.set_ylabel('Total Cost', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Set x-axis to show only research values
    ax.set_xticks([15, 30, 50])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    return fig

def create_cost_efficiency_scalability(df):
    """Cost per Customer vs Number of Customers (Average only, no variance)"""
    if df.empty or df['num_customers_rounded'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data using rounded customer counts and only get mean
    scalability_data = df.groupby(['algorithm', 'num_customers_rounded']).agg({
        'cost_per_customer': 'mean'
    }).reset_index()
    
    scalability_data.columns = ['algorithm', 'num_customers_rounded', 'cost_per_cust_mean']
    
    algorithms = scalability_data['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = scalability_data[scalability_data['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_customers_rounded'], alg_data['cost_per_cust_mean'], 
                   label=alg, marker='^', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Cost Efficiency: Cost per Customer vs Number of Customers', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Number of Customers', fontsize=14)
    ax.set_ylabel('Cost per Customer', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set more detailed y-axis ticks to show small differences
    y_min, y_max = ax.get_ylim()
    if y_max - y_min < 1000000:  # If range is less than 1M, show more detail
        # Create more detailed ticks
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Set x-axis to show only research values
    ax.set_xticks([15, 30, 50])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    return fig

def create_time_efficiency_scalability(df):
    """Time per Customer vs Number of Customers (Average only, no variance)"""
    if df.empty or df['num_customers_rounded'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data using rounded customer counts and only get mean
    scalability_data = df.groupby(['algorithm', 'num_customers_rounded']).agg({
        'time_per_customer': 'mean'
    }).reset_index()
    
    scalability_data.columns = ['algorithm', 'num_customers_rounded', 'time_per_cust_mean']
    
    algorithms = scalability_data['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = scalability_data[scalability_data['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_customers_rounded'], alg_data['time_per_cust_mean'], 
                   label=alg, marker='d', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Time Efficiency: Time per Customer vs Number of Customers', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Number of Customers', fontsize=14)
    ax.set_ylabel('Time per Customer (seconds)', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show only research values
    ax.set_xticks([15, 30, 50])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    return fig

# ============= CARGO MIX ANALYSIS (3 separate plots) =============

def create_cargo_cost_distribution(df):
    """Cost Distribution by Cargo Mix (Item Ratios) - Multipanel per Algorithm"""
    if df.empty or df['cargo_mix'].isna().all():
        return None
    
    df_clean = df[df['cargo_mix'] != 'Unknown'].copy()
    if df_clean.empty:
        return None
    
    # Set up the subplot grid
    algorithms = sorted(df_clean['algorithm'].unique())
    n_algs = len(algorithms)
    
    # Determine grid layout
    if n_algs <= 2:
        rows, cols = 1, n_algs
        figsize = (7 * n_algs, 6)
    else:
        rows, cols = 2, 2
        figsize = (14, 12)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle case where there's only one subplot
    if n_algs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    # Create plot for each algorithm
    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        alg_data = df_clean[df_clean['algorithm'] == alg]
        
        if not alg_data.empty:
            # Create boxplot using seaborn
            sns.boxplot(data=alg_data, x='cargo_mix', y='total_cost', ax=ax, palette='Set3')
            
            ax.set_title(f'{alg} Algorithm', fontweight='bold', fontsize=14)
            ax.set_xlabel('', fontsize=12)
            ax.set_ylabel('Total Cost', fontsize=12)
            ax.tick_params(axis='x', rotation=0, labelsize=10)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
            
            # Crop y-axis if needed
            y_values = alg_data['total_cost'].dropna()
            if len(y_values) > 0:
                y_min, y_max = y_values.min(), y_values.max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(bottom=max(0, y_min - y_range * 0.1))
    
    # Hide extra subplots if any
    for idx in range(n_algs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Total Cost Distribution by Item Ratio Configuration', 
                fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    return fig

def create_cargo_time_distribution(df):
    """Running Time Distribution by Cargo Mix (Item Ratios) - Multipanel per Algorithm"""
    if df.empty or df['cargo_mix'].isna().all():
        return None
    
    df_clean = df[df['cargo_mix'] != 'Unknown'].copy()
    if df_clean.empty:
        return None
    
    # Set up the subplot grid
    algorithms = sorted(df_clean['algorithm'].unique())
    n_algs = len(algorithms)
    
    # Determine grid layout
    if n_algs <= 2:
        rows, cols = 1, n_algs
        figsize = (7 * n_algs, 6)
    else:
        rows, cols = 2, 2
        figsize = (14, 12)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle case where there's only one subplot
    if n_algs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    # Create plot for each algorithm
    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        alg_data = df_clean[df_clean['algorithm'] == alg]
        
        if not alg_data.empty:
            # Create boxplot using seaborn
            sns.boxplot(data=alg_data, x='cargo_mix', y='running_time', ax=ax, palette='Set2')
            
            ax.set_title(f'{alg} Algorithm', fontweight='bold', fontsize=14)
            ax.set_xlabel('', fontsize=12)
            ax.set_ylabel('Running Time (seconds)', fontsize=12)
            ax.tick_params(axis='x', rotation=0, labelsize=10)
            ax.grid(True, alpha=0.3)
            
            # Crop y-axis if needed
            y_values = alg_data['running_time'].dropna()
            if len(y_values) > 0:
                y_min, y_max = y_values.min(), y_values.max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(bottom=max(0, y_min - y_range * 0.1))
    
    # Hide extra subplots if any
    for idx in range(n_algs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Running Time Distribution by Item Ratio Configuration', 
                fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    return fig

def create_cargo_algorithm_comparison(df):
    """Algorithm Performance Comparison by Item Ratios"""
    if df.empty or df['cargo_mix'].isna().all():
        return None
    
    df_clean = df[df['cargo_mix'] != 'Unknown'].copy()
    if df_clean.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    efficiency_analysis = df_clean.groupby(['cargo_mix', 'algorithm']).agg({
        'cost_per_customer': 'mean'
    }).reset_index()
    
    cargo_types_sorted = sorted(efficiency_analysis['cargo_mix'].unique())
    algorithms = sorted(efficiency_analysis['algorithm'].unique())
    
    x = np.arange(len(cargo_types_sorted))
    width = 0.2
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    all_costs = []
    for i, alg in enumerate(algorithms):
        alg_data = efficiency_analysis[efficiency_analysis['algorithm'] == alg]
        
        costs_by_cargo = []
        for cargo in cargo_types_sorted:
            cargo_data = alg_data[alg_data['cargo_mix'] == cargo]
            if not cargo_data.empty:
                cost_val = cargo_data['cost_per_customer'].iloc[0]
                costs_by_cargo.append(cost_val)
                all_costs.append(cost_val)
            else:
                costs_by_cargo.append(0)
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.bar(x + offset, costs_by_cargo, width, label=alg, color=colors[i], alpha=0.8)
    
    # Crop y-axis to highlight differences
    if all_costs:
        min_cost = min([c for c in all_costs if c > 0])
        max_cost = max(all_costs)
        y_range = max_cost - min_cost
        y_min = min_cost - (y_range * 0.1)  # Start 10% below minimum
        ax.set_ylim(bottom=max(0, y_min))
    
    ax.set_title('Cost Efficiency Comparison by Item Ratio Configuration', 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Cost per Customer', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cargo_types_sorted, rotation=0, fontsize=11, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    return fig

# ============= CLUSTER ANALYSIS =============

def create_cluster_cost_analysis(df):
    """Cost Analysis by Number of Clusters (Average only, no variance)"""
    if df.empty or df['num_clusters'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cluster_analysis = df.groupby(['algorithm', 'num_clusters']).agg({
        'total_cost': 'mean'
    }).reset_index()
    
    cluster_analysis.columns = ['algorithm', 'num_clusters', 'cost_mean']
    
    algorithms = cluster_analysis['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = cluster_analysis[cluster_analysis['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_clusters'], alg_data['cost_mean'],
                   label=alg, marker='o', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Total Cost vs Number of Customer Clusters', fontweight='bold', fontsize=16)
    ax.set_xlabel('Number of Customer Clusters', fontsize=14)
    ax.set_ylabel('Total Cost', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    return fig

def create_cluster_time_analysis(df):
    """Time Analysis by Number of Clusters (Average only, no variance)"""
    if df.empty or df['num_clusters'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cluster_analysis = df.groupby(['algorithm', 'num_clusters']).agg({
        'running_time': 'mean'
    }).reset_index()
    
    cluster_analysis.columns = ['algorithm', 'num_clusters', 'time_mean']
    
    algorithms = cluster_analysis['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    for i, alg in enumerate(algorithms):
        alg_data = cluster_analysis[cluster_analysis['algorithm'] == alg]
        if not alg_data.empty:
            ax.plot(alg_data['num_clusters'], alg_data['time_mean'],
                   label=alg, marker='s', color=colors[i], linewidth=3, markersize=8)
    
    ax.set_title('Running Time vs Number of Customer Clusters', fontweight='bold', fontsize=16)
    ax.set_xlabel('Number of Customer Clusters', fontsize=14)
    ax.set_ylabel('Running Time (seconds)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============= LOCATION COMPARISON (3 separate plots) =============

def create_location_cost_comparison(df):
    """Average Total Cost by Location and Algorithm"""
    if df.empty or df['location'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    locations = df['location'].dropna().unique()
    algorithms = df['algorithm'].unique()
    
    location_cost = df.groupby(['location', 'algorithm'])['total_cost'].mean().unstack(fill_value=0)
    
    if not location_cost.empty:
        x = np.arange(len(locations))
        width = 0.2
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        all_costs = []
        for i, alg in enumerate(algorithms):
            if alg in location_cost.columns:
                costs = [location_cost.loc[loc, alg] if loc in location_cost.index else 0 for loc in locations]
                all_costs.extend([c for c in costs if c > 0])
                offset = (i - len(algorithms)/2 + 0.5) * width
                ax.bar(x + offset, costs, width, label=alg, color=colors[i], alpha=0.8)
        
        # Crop y-axis to highlight differences
        if all_costs:
            min_cost = min(all_costs)
            max_cost = max(all_costs)
            y_range = max_cost - min_cost
            y_min = min_cost - (y_range * 0.15)  # Start 15% below minimum
            ax.set_ylim(bottom=max(0, y_min))
        
        ax.set_title('Average Total Cost by Location and Algorithm', fontweight='bold', fontsize=16)
        ax.set_xlabel('Location', fontsize=14)
        ax.set_ylabel('Average Total Cost', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    return fig

def create_location_time_comparison(df):
    """Average Running Time by Location and Algorithm"""
    if df.empty or df['location'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    locations = df['location'].dropna().unique()
    algorithms = df['algorithm'].unique()
    
    location_time = df.groupby(['location', 'algorithm'])['running_time'].mean().unstack(fill_value=0)
    
    if not location_time.empty:
        x = np.arange(len(locations))
        width = 0.2
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        all_times = []
        for i, alg in enumerate(algorithms):
            if alg in location_time.columns:
                times = [location_time.loc[loc, alg] if loc in location_time.index else 0 for loc in locations]
                all_times.extend([t for t in times if t > 0])
                offset = (i - len(algorithms)/2 + 0.5) * width
                ax.bar(x + offset, times, width, label=alg, color=colors[i], alpha=0.8)
        
        # Crop y-axis to highlight differences
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            y_range = max_time - min_time
            y_min = min_time - (y_range * 0.2)  # Start 20% below minimum
            ax.set_ylim(bottom=max(0, y_min))
        
        ax.set_title('Average Running Time by Location and Algorithm', fontweight='bold', fontsize=16)
        ax.set_xlabel('Location', fontsize=14)
        ax.set_ylabel('Average Running Time (seconds)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_location_efficiency_comparison(df):
    """Cost Efficiency by Location"""
    if df.empty or df['location'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    locations = df['location'].dropna().unique()
    efficiency_data = []
    efficiency_labels = []
    
    for loc in locations:
        loc_data = df[df['location'] == loc]['cost_per_customer'].dropna()
        if not loc_data.empty:
            efficiency_data.append(loc_data.values)
            efficiency_labels.append(loc)
    
    if efficiency_data:
        bp = ax.boxplot(efficiency_data, labels=efficiency_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(efficiency_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Cost Efficiency Distribution by Location', fontweight='bold', fontsize=16)
        ax.set_xlabel('Location', fontsize=14)
        ax.set_ylabel('Cost per Customer', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    return fig

# ============= DATA TYPE COMPARISON =============

def create_data_type_comparison(df):
    """Historical vs Generated Data Comparison"""
    if df.empty or df['data_type'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data_types = df['data_type'].dropna().unique()
    algorithms = df['algorithm'].unique()
    
    datatype_cost = df.groupby(['data_type', 'algorithm'])['total_cost'].mean().unstack(fill_value=0)
    
    if not datatype_cost.empty:
        x = np.arange(len(data_types))
        width = 0.2
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        all_costs = []
        for i, alg in enumerate(algorithms):
            if alg in datatype_cost.columns:
                costs = [datatype_cost.loc[dt, alg] if dt in datatype_cost.index else 0 for dt in data_types]
                all_costs.extend([c for c in costs if c > 0])
                offset = (i - len(algorithms)/2 + 0.5) * width
                ax.bar(x + offset, costs, width, label=alg, color=colors[i], alpha=0.8)
        
        # Crop y-axis to highlight differences
        if all_costs:
            min_cost = min(all_costs)
            max_cost = max(all_costs)
            y_range = max_cost - min_cost
            y_min = min_cost - (y_range * 0.15)  # Start 15% below minimum
            ax.set_ylim(bottom=max(0, y_min))
        
        ax.set_title('Average Total Cost: Historical vs Generated Data', fontweight='bold', fontsize=16)
        ax.set_xlabel('Data Type', fontsize=14)
        ax.set_ylabel('Average Total Cost', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([dt.title() for dt in data_types], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the enhanced analysis"""
    base_path = os.getcwd()
    
    print("🔍 Enhanced Algorithm Performance Analysis (No Variance + Customer Rounding)")
    print("=" * 80)
    print(f"Looking for CSV files in: {base_path}/results/")
    print("Expected filename pattern: LOCATION_nc_X_ncl_X_nv_X_DATATYPE_r_X_X_X")
    print("Example: JK2_nc_15_ncl_5_nv_30_generated_r_0.20_0.20_0.60")
    print("📊 Customer counts will be rounded to research values: 15, 30, 50")
    print("-" * 80)
    
    # Load enhanced data
    df = load_enhanced_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure and naming convention.")
        return
    
    print(f"\n📈 Creating focused visualizations (average only, customer rounding)...")
    
    all_figures = []
    
    # ===== SCALABILITY ANALYSIS =====
    print("\n🔍 SCALABILITY ANALYSIS")
    print("1. Running time vs customers (average only)...")
    fig1 = create_running_time_scalability(df)
    if fig1:
        all_figures.append((fig1, "01_running_time_scalability"))
        plt.show()
    
    print("2. Total cost vs customers (average only)...")
    fig2 = create_total_cost_scalability(df)
    if fig2:
        all_figures.append((fig2, "02_total_cost_scalability"))
        plt.show()
    
    print("3. Cost efficiency vs customers (average only)...")
    fig3 = create_cost_efficiency_scalability(df)
    if fig3:
        all_figures.append((fig3, "03_cost_efficiency_scalability"))
        plt.show()
    
    print("4. Time efficiency vs customers (average only)...")
    fig4 = create_time_efficiency_scalability(df)
    if fig4:
        all_figures.append((fig4, "04_time_efficiency_scalability"))
        plt.show()
    
    # ===== ITEM RATIO ANALYSIS =====
    print("\n📦 ITEM RATIO ANALYSIS")
    print("5. Cost distribution by item ratios...")
    fig5 = create_cargo_cost_distribution(df)
    if fig5:
        all_figures.append((fig5, "05_item_ratio_cost_distribution"))
        plt.show()
    
    print("6. Time distribution by item ratios...")
    fig6 = create_cargo_time_distribution(df)
    if fig6:
        all_figures.append((fig6, "06_item_ratio_time_distribution"))
        plt.show()
    
    print("7. Algorithm comparison by item ratios...")
    fig7 = create_cargo_algorithm_comparison(df)
    if fig7:
        all_figures.append((fig7, "07_item_ratio_algorithm_comparison"))
        plt.show()
    
    # ===== CLUSTER ANALYSIS =====
    print("\n🔢 CLUSTER ANALYSIS")
    print("8. Cost vs number of clusters (average only)...")
    fig8 = create_cluster_cost_analysis(df)
    if fig8:
        all_figures.append((fig8, "08_cluster_cost_analysis"))
        plt.show()
    
    print("9. Time vs number of clusters (average only)...")
    fig9 = create_cluster_time_analysis(df)
    if fig9:
        all_figures.append((fig9, "09_cluster_time_analysis"))
        plt.show()
    
    # ===== LOCATION COMPARISON =====
    print("\n📍 LOCATION COMPARISON")
    print("10. Location cost comparison...")
    fig10 = create_location_cost_comparison(df)
    if fig10:
        all_figures.append((fig10, "10_location_cost_comparison"))
        plt.show()
    
    print("11. Location time comparison...")
    fig11 = create_location_time_comparison(df)
    if fig11:
        all_figures.append((fig11, "11_location_time_comparison"))
        plt.show()
    
    print("12. Location efficiency comparison...")
    fig12 = create_location_efficiency_comparison(df)
    if fig12:
        all_figures.append((fig12, "12_location_efficiency_comparison"))
        plt.show()
    
    # ===== DATA TYPE COMPARISON =====
    print("\n📊 DATA TYPE COMPARISON")
    print("13. Historical vs Generated data...")
    fig13 = create_data_type_comparison(df)
    if fig13:
        all_figures.append((fig13, "13_data_type_comparison"))
        plt.show()
    
    # Print key insights
    print("\n" + "=" * 80)
    print("📊 KEY INSIGHTS SUMMARY")
    print("=" * 80)
    
    if not df.empty:
        # Best performing algorithm by cost
        best_cost_alg = df.groupby('algorithm')['total_cost'].mean().idxmin()
        print(f"🏆 Best Average Cost: {best_cost_alg}")
        
        # Best performing algorithm by time
        best_time_alg = df.groupby('algorithm')['running_time'].mean().idxmin()
        print(f"⚡ Fastest Algorithm: {best_time_alg}")
        
        # Most efficient algorithm (cost per customer)
        best_efficiency_alg = df.groupby('algorithm')['cost_per_customer'].mean().idxmin()
        print(f"💰 Most Cost Efficient: {best_efficiency_alg}")
        
        # Scalability insights using rounded customer counts
        scalability_scores = df.groupby('algorithm').apply(
            lambda x: x['running_time'].corr(x['num_customers_rounded']) if len(x) > 1 else 0
        ).sort_values()
        best_scalability_alg = scalability_scores.index[0]
        print(f"📈 Best Scalability: {best_scalability_alg}")
        
        print("\n📋 Algorithm Rankings:")
        cost_ranking = df.groupby('algorithm')['total_cost'].mean().sort_values()
        for i, (alg, cost) in enumerate(cost_ranking.items(), 1):
            print(f"  {i}. {alg}: {cost:,.0f} average cost")
        
        print(f"\n📊 Customer Count Mapping:")
        customer_mapping = df.groupby(['num_customers', 'num_customers_rounded']).size().reset_index()
        customer_mapping = customer_mapping[['num_customers', 'num_customers_rounded']].drop_duplicates().sort_values('num_customers')
        for _, row in customer_mapping.iterrows():
            print(f"  {row['num_customers']} → {row['num_customers_rounded']}")
    
    # Save option
    print(f"\n💾 Generated {len(all_figures)} focused visualizations")
    save_plots = input("Do you want to save all plots? (y/n): ").lower().strip()
    
    if save_plots == 'y':
        print("\nSaving plots...")
        for fig, name in all_figures:
            filename = f'{name}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved {filename}")
        
        print(f"\n🎉 Successfully saved {len(all_figures)} focused visualizations!")
        print("\n📁 Files created (with customer rounding & no variance):")
        print("  📈 SCALABILITY ANALYSIS:")
        print("    01_running_time_scalability.png")
        print("    02_total_cost_scalability.png") 
        print("    03_cost_efficiency_scalability.png")
        print("    04_time_efficiency_scalability.png")
        print("  📦 ITEM RATIO ANALYSIS:")
        print("    05_item_ratio_cost_distribution.png")
        print("    06_item_ratio_time_distribution.png")
        print("    07_item_ratio_algorithm_comparison.png")
        print("  🔢 CLUSTER ANALYSIS:")
        print("    08_cluster_cost_analysis.png")
        print("    09_cluster_time_analysis.png")
        print("  📍 LOCATION COMPARISON:")
        print("    10_location_cost_comparison.png")
        print("    11_location_time_comparison.png")
        print("    12_location_efficiency_comparison.png")
        print("  📊 DATA TYPE & SUMMARY:")
        print("    13_data_type_comparison.png")

if __name__ == "__main__":
    main()