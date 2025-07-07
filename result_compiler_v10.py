import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
import json
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
    Example: JK2_nc_30_ncl_3_nv_30_historical_r_0.20_0.20_0.60
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

def read_instance_file(instance_path):
    """
    Read instance JSON file and extract reefer information
    Expected format: JSON with customers and their items containing is_reefer_required field
    """
    try:
        with open(instance_path, 'r') as file:
            instance_data = json.load(file)
        
        reefer_count = 0
        total_items = 0
        
        # Process each customer and their items
        customers = instance_data.get('customers', {})
        for customer_id, customer_data in customers.items():
            items = customer_data.get('items', [])
            for item in items:
                total_items += 1
                if item.get('is_reefer_required', False):
                    reefer_count += 1
        
        if total_items == 0:
            return None
        
        reefer_percent = (reefer_count / total_items) * 100
        non_reefer_percent = 100 - reefer_percent
        
        return {
            'reefer_percent': reefer_percent,
            'non_reefer_percent': non_reefer_percent,
            'total_items': total_items,
            'reefer_count': reefer_count,
            'total_customers': len(customers)
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {instance_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading instance file {instance_path}: {e}")
        return None

def calculate_reefer_percentage_from_instance(instance_filename, base_path):
    """
    Calculate reefer percentage by reading the corresponding instance JSON file
    """
    instances_path = os.path.join(base_path, 'instances')
    
    if not os.path.exists(instances_path):
        print(f"Instances folder not found at: {instances_path}")
        return None
    
    # Try to find matching instance file
    instance_file_path = os.path.join(instances_path, instance_filename)
    
    if os.path.exists(instance_file_path):
        return read_instance_file(instance_file_path)
    else:
        # Try without .csv extension and add .json extension
        base_name = os.path.splitext(instance_filename)[0]
        possible_extensions = ['.json', '.txt', '.dat', '']
        
        for ext in possible_extensions:
            test_path = os.path.join(instances_path, base_name + ext)
            if os.path.exists(test_path):
                return read_instance_file(test_path)
        
        print(f"Instance file not found for: {instance_filename}")
        return None

def load_zone_data(base_path):
    """
    Load CSV files and organize by multiple enhanced dimensions
    Connect with instance files to get real reefer information
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
            
            # Get reefer information from corresponding instance file
            reefer_info = calculate_reefer_percentage_from_instance(filename, base_path)
            
            try:
                # Read CSV data
                df = pd.read_csv(file_path, header=None, names=['total_cost', 'running_time'])
                
                # Add metadata to each row
                for _, row in df.iterrows():
                    record = {
                        'algorithm': algorithm_name,
                        'location': location,
                        'num_customers': num_customers,
                        'num_customers_rounded': round_customer_count(num_customers),
                        'num_clusters': num_clusters,
                        'num_vehicles': num_vehicles,
                        'data_type': data_type,
                        'ratio_small': ratios[0] if ratios else None,
                        'ratio_medium': ratios[1] if ratios else None,
                        'ratio_large': ratios[2] if ratios else None,
                        'reefer_percent': reefer_info['reefer_percent'] if reefer_info else None,
                        'non_reefer_percent': reefer_info['non_reefer_percent'] if reefer_info else None,
                        'reefer_count': reefer_info['reefer_count'] if reefer_info else None,
                        'total_items_instance': reefer_info['total_items'] if reefer_info else None,
                        'total_customers_instance': reefer_info['total_customers'] if reefer_info else None,
                        'total_cost': row['total_cost'],
                        'running_time': row['running_time'],
                        'filename': filename
                    }
                    enhanced_data.append(record)
                
                reefer_status = f"({reefer_info['reefer_percent']:.1f}% reefer, {reefer_info['reefer_count']}/{reefer_info['total_items']} items)" if reefer_info else "(no reefer data)"
                print(f"✓ Processed {filename} -> {algorithm_name}, {location}, nc:{num_customers}→{round_customer_count(num_customers)}, ncl:{num_clusters}, {data_type} {reefer_status}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    
    return df

def create_3cluster_30customers_analysis(df):
    """
    Analysis for 3 clusters, 30 customers, historical data, JK2 depot with reefer information
    """
    # Filter data according to specifications
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 30) & 
        (df['num_clusters'] == 3) &
        (df['location'] == 'JK2')
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for JK2 depot, 3 clusters, 30 customers, historical data")
        return None
    
    print(f"📊 JK2 Depot - 3 Clusters, 30 Customers Analysis - Found {len(filtered_df)} records:")
    print(f"  - Depot: JK2")
    print(f"  - Data type: historical")
    print(f"  - Customers: 30 (rounded)")
    print(f"  - Clusters: 3")
    print(f"  - Algorithms: {sorted(filtered_df['algorithm'].unique())}")
    
    # Get unique problem instances
    problem_instances = filtered_df.groupby(['algorithm', 'reefer_percent', 'non_reefer_percent']).agg({
        'total_cost': 'mean'
    }).reset_index()
    
    # Create the visualization - only total cost
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for plotting
    problem_instances['instance_label'] = problem_instances.apply(
        lambda row: f"{row['algorithm']}\n({row['reefer_percent']:.0f}% Reefer)" if pd.notna(row['reefer_percent']) 
                   else f"{row['algorithm']}\n(No Reefer Data)", axis=1
    )
    
    # Sort by total cost for better visualization
    problem_instances_sorted = problem_instances.sort_values('total_cost')
    
    # Plot: Total Cost Analysis
    x_pos = np.arange(len(problem_instances_sorted))
    colors = plt.cm.plasma(np.linspace(0, 1, len(problem_instances_sorted)))
    
    bars = ax.bar(x_pos, problem_instances_sorted['total_cost'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add reefer percentage labels on bars
    for i, (bar, row) in enumerate(zip(bars, problem_instances_sorted.itertuples())):
        height = bar.get_height()
        if pd.notna(row.reefer_percent):
            reefer_label = f'{row.reefer_percent:.0f}%R\n{row.non_reefer_percent:.0f}%NR'
        else:
            reefer_label = 'No Reefer\nData'
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                reefer_label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Total Cost Analysis by Problem Instance\n(JK2 Depot, 3 Clusters, 30 Customers, Historical Data)', 
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Problem Instance (Algorithm)', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(problem_instances_sorted['instance_label'], rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, problem_instances_sorted

def create_algorithm_comparison_chart(df):
    """
    Create GA algorithm performance vs reefer ratio chart for MKS depot
    """
    # Filter data according to specifications
    filtered_df = df[
        (df['data_type'].str.lower() == 'historical') & 
        (df['num_customers_rounded'] == 30) & 
        (df['num_clusters'] == 3) &
        (df['location'] == 'MKS') &
        (df['algorithm'] == 'GA')
    ].copy()
    
    if filtered_df.empty:
        print("⚠️ No data found for MKS depot GA algorithm analysis")
        return None, None
    
    # Group by reefer percentage for GA algorithm
    ga_stats = filtered_df.groupby('reefer_percent').agg({
        'total_cost': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    ga_stats.columns = ['reefer_percent', 'cost_mean', 'cost_std', 'count']
    
    # Remove rows with NaN reefer_percent
    ga_stats = ga_stats.dropna(subset=['reefer_percent'])
    
    if ga_stats.empty:
        print("⚠️ No valid reefer data found for GA algorithm comparison")
        return None, None
    
    # Create comparison chart - GA performance across different reefer ratios
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Sort by reefer percentage
    ga_stats_sorted = ga_stats.sort_values('reefer_percent')
    
    # Create bar chart
    x_pos = np.arange(len(ga_stats_sorted))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ga_stats_sorted)))
    
    bars = ax.bar(x_pos, ga_stats_sorted['cost_mean'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add error bars if standard deviation is available (no text labels)
    if not ga_stats_sorted['cost_std'].isna().all():
        ax.errorbar(x_pos, ga_stats_sorted['cost_mean'], 
                   yerr=ga_stats_sorted['cost_std'], fmt='none', 
                   color='black', capsize=5, alpha=0.7)
    
    ax.set_title('GA Algorithm Performance vs Reefer Ratio\n(MKS Depot, 3 Clusters, 30 Customers, Historical)', 
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Reefer Percentage', fontsize=12)
    ax.set_ylabel('Average Total Cost', fontsize=12)
    
    # Set x-axis labels to show reefer percentages
    reefer_labels = [f'{reefer:.1f}%' for reefer in ga_stats_sorted['reefer_percent']]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(reefer_labels, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis with appropriate suffixes
    def format_currency(x, pos):
        if abs(x) >= 1e9:
            return f'{x/1e9:.1f}B'
        elif abs(x) >= 1e6:
            return f'{x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
    
    plt.tight_layout()
    return fig, ga_stats_sorted

def print_detailed_insights(df, problem_instances):
    """
    Print detailed insights for MKS depot GA algorithm analysis
    """
    print("\n" + "=" * 90)
    print("📊 DETAILED INSIGHTS - MKS DEPOT, GA ALGORITHM, 3 CLUSTERS, 30 CUSTOMERS, HISTORICAL DATA")
    print("=" * 90)
    
    # Verify all instances are GA
    if 'algorithm' in problem_instances.columns:
        unique_algorithms = problem_instances['algorithm'].unique()
        if len(unique_algorithms) > 1 or (len(unique_algorithms) == 1 and unique_algorithms[0] != 'GA'):
            print(f"⚠️ Warning: Expected only GA algorithm, but found: {unique_algorithms}")
    
    # Problem instances summary
    print(f"\n🔍 GA Problem Instances Found: {len(problem_instances)}")
    print("-" * 50)
    
    for idx, row in problem_instances.iterrows():
        print(f"\n📋 GA Instance {idx+1}:")
        if pd.notna(row['reefer_percent']):
            print(f"   🚛 Reefer goods: {row['reefer_percent']:.1f}%")
            print(f"   📦 Non-reefer goods: {row['non_reefer_percent']:.1f}%")
        else:
            print(f"   ⚠️ Reefer data not available")
        print(f"   💰 Total cost: {row['total_cost']:.2f}")
    
    # Check if we have reefer data
    has_reefer_data = problem_instances['reefer_percent'].notna().any()
    
    # Best and worst performers
    print(f"\n🏆 GA ALGORITHM PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    best_cost_idx = problem_instances['total_cost'].idxmin()
    worst_cost_idx = problem_instances['total_cost'].idxmax()
    
    print(f"💰 Best GA Instance (Lowest Cost):")
    print(f"   Cost: {problem_instances.loc[best_cost_idx, 'total_cost']:.2f}")
    if has_reefer_data and pd.notna(problem_instances.loc[best_cost_idx, 'reefer_percent']):
        print(f"   Reefer: {problem_instances.loc[best_cost_idx, 'reefer_percent']:.1f}%")
    
    if len(problem_instances) > 1:
        print(f"\n💸 Worst GA Instance (Highest Cost):")
        print(f"   Cost: {problem_instances.loc[worst_cost_idx, 'total_cost']:.2f}")
        if has_reefer_data and pd.notna(problem_instances.loc[worst_cost_idx, 'reefer_percent']):
            print(f"   Reefer: {problem_instances.loc[worst_cost_idx, 'reefer_percent']:.1f}%")
        
        # Cost difference analysis
        cost_diff = problem_instances.loc[worst_cost_idx, 'total_cost'] - problem_instances.loc[best_cost_idx, 'total_cost']
        cost_diff_pct = (cost_diff / problem_instances.loc[best_cost_idx, 'total_cost']) * 100
        print(f"\n📊 Cost Variation: {cost_diff:.2f} ({cost_diff_pct:.1f}% difference)")
    
    # Reefer ratio analysis (only if we have reefer data)
    if has_reefer_data:
        print(f"\n🧊 REEFER RATIO ANALYSIS FOR GA ALGORITHM")
        print("-" * 45)
        valid_reefer_data = problem_instances.dropna(subset=['reefer_percent'])
        
        if not valid_reefer_data.empty:
            unique_reefer_ratios = valid_reefer_data['reefer_percent'].unique()
            print(f"📊 Unique reefer ratios tested with GA: {sorted(unique_reefer_ratios)}")
            
            for ratio in sorted(unique_reefer_ratios):
                instances_with_ratio = valid_reefer_data[valid_reefer_data['reefer_percent'] == ratio]
                avg_cost = instances_with_ratio['total_cost'].mean()
                print(f"   {ratio:.1f}% reefer → Cost: {avg_cost:.2f}")
                
            # Correlation analysis
            if len(unique_reefer_ratios) > 1:
                correlation = valid_reefer_data['reefer_percent'].corr(valid_reefer_data['total_cost'])
                print(f"\n📈 GA Performance vs Reefer Correlation: {correlation:.3f}")
                if correlation > 0.5:
                    print("   → Strong positive correlation: Higher reefer % → Higher cost for GA")
                elif correlation < -0.5:
                    print("   → Strong negative correlation: Higher reefer % → Lower cost for GA")
                else:
                    print("   → Weak correlation between reefer % and cost for GA")
        else:
            print("   No valid reefer data available for GA analysis")
    else:
        print(f"\n⚠️ No reefer data available - check instances folder and file format")
    
    # Instance ranking
    print(f"\n🏅 GA INSTANCE RANKING (by cost):")
    print("-" * 35)
    sorted_by_cost = problem_instances.sort_values('total_cost')
    for i, (idx, row) in enumerate(sorted_by_cost.iterrows(), 1):
        reefer_info = f"({row['reefer_percent']:.1f}% reefer)" if pd.notna(row['reefer_percent']) else "(no reefer data)"
        print(f"   {i}. GA Instance: {row['total_cost']:.2f} {reefer_info}")
    
    # Data source information
    print(f"\n📁 DATA SOURCE INFORMATION")
    print("-" * 35)
    print(f"   Algorithm focus: GA (Genetic Algorithm) only")
    print(f"   Depot location: MKS")
    print(f"   Instance files source: instances/ folder (JSON format)")
    print(f"   Reefer calculation: Based on is_reefer_required field in items")
    print(f"   Reefer data availability: {has_reefer_data}")
    if has_reefer_data:
        valid_count = problem_instances['reefer_percent'].notna().sum()
        total_count = len(problem_instances)
        print(f"   Valid reefer data: {valid_count}/{total_count} GA instances")

def main():
    """Main function for 3 clusters, 30 customers analysis"""
    base_path = os.getcwd()
    
    print("🔍 3 Clusters, 30 Customers Analysis with Reefer Information")
    print("=" * 65)
    print("Filtering criteria:")
    print("  - Data type: Historical")
    print("  - Number of customers: 30 (rounded)")
    print("  - Number of clusters: 3")
    print("  - Include reefer percentage information")
    print("=" * 65)
    
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
    
    # Create main analysis
    print("\n📈 Creating 3 clusters, 30 customers analysis...")
    fig_main, problem_instances = create_3cluster_30customers_analysis(df)
    
    if fig_main is None:
        print("❌ No suitable data found for analysis")
        return
    
    plt.show()
    
    # Create algorithm comparison
    print("\n📈 Creating algorithm comparison chart...")
    fig_algo, algorithm_stats = create_algorithm_comparison_chart(df)
    
    if fig_algo:
        plt.show()
    
    # Print detailed insights
    print_detailed_insights(df, problem_instances)
    
    # Save option
    if fig_main or fig_algo:
        save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
        
        if save_plots == 'y':
            if fig_main:
                fig_main.savefig('3clusters_30customers_analysis.png', dpi=300, bbox_inches='tight')
                print("✅ Saved 3clusters_30customers_analysis.png")
            
            if fig_algo:
                fig_algo.savefig('algorithm_comparison_3clusters_30customers.png', dpi=300, bbox_inches='tight')
                print("✅ Saved algorithm_comparison_3clusters_30customers.png")
            
            print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()