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
                        'num_customers_rounded': round_customer_count(num_customers),
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
                
                print(f"✓ Processed {filename}")
                
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

def create_running_time_cluster_analysis(df):
    """
    Create multipanel plot for running time analysis: 
    Left panel shows algorithm comparison, Right panel shows cluster averages
    """
    if df.empty or df['num_clusters'].isna().all():
        print("No data available for cluster analysis")
        return None
    
    # Clean data - remove missing values
    df_clean = df.dropna(subset=['num_clusters', 'algorithm', 'running_time']).copy()
    
    if df_clean.empty:
        print("No clean data available")
        return None
    
    print(f"📊 Creating running time cluster analysis:")
    print(f"  - Cluster counts: {sorted(df_clean['num_clusters'].unique())}")
    print(f"  - Algorithms: {sorted(df_clean['algorithm'].unique())}")
    print(f"  - Customer counts (rounded): {sorted(df_clean['num_customers_rounded'].dropna().unique())}")
    print(f"  - Total records: {len(df_clean)}")
    
    # Get unique clusters and algorithms
    clusters = sorted(df_clean['num_clusters'].unique())
    algorithms = sorted(df_clean['algorithm'].unique())
    
    print(f"📊 Creating multipanel plot with {len(clusters)} clusters and {len(algorithms)} algorithms")
    
    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Set colors to match previous plots
    algorithm_colors = {
        'BRKGA': '#1f77b4',  # Blue
        'DE': '#d62728',     # Red  
        'GA': '#e377c2',     # Pink
        'PSO': '#17becf'     # Cyan
    }
    
    # === LEFT PANEL: ALGORITHM COMPARISON BY CLUSTER COUNT ===
    print("📊 Creating left panel: Algorithm comparison for running time")
    cluster_analysis = df_clean.groupby(['algorithm', 'num_clusters']).agg({
        'running_time': 'mean'
    }).reset_index()
    
    cluster_analysis.columns = ['algorithm', 'num_clusters', 'avg_time']
    
    # Set up bar chart
    x1 = np.arange(len(clusters))
    width = 0.2
    
    # Create bars for each algorithm (LEFT PANEL)
    for i, alg in enumerate(algorithms):
        alg_data = cluster_analysis[cluster_analysis['algorithm'] == alg]
        
        times = []
        for cluster in clusters:
            cluster_data = alg_data[alg_data['num_clusters'] == cluster]
            if not cluster_data.empty:
                times.append(cluster_data['avg_time'].iloc[0])
            else:
                times.append(0)
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        color = algorithm_colors.get(alg, f'C{i}')
        ax1.bar(x1 + offset, times, width, label=alg, color=color, alpha=0.8)
        print(f"   Added {alg} with times: {times}")
    
    # Customize LEFT PANEL
    ax1.set_title('Algorithm Performance Across Cluster Counts', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Cluster Count', fontsize=12)
    ax1.set_ylabel('Running Time (seconds)', fontsize=12)
    
    # Set x-axis labels
    cluster_labels = [f'{int(cluster)} Cluster{"s" if cluster != 1 else ""}' for cluster in clusters]
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cluster_labels, fontsize=11)
    
    # Format y-axis
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    
    # Add legend and grid
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # === RIGHT PANEL: AVERAGE RUNNING TIME BY CLUSTER COUNT (ACROSS ALL ALGORITHMS) ===
    print("📊 Creating right panel: Cluster averages for running time")
    # Calculate average running time per cluster count (averaged across all algorithms)
    cluster_avg = df_clean.groupby('num_clusters').agg({
        'running_time': 'mean'
    }).reset_index()
    
    cluster_avg.columns = ['num_clusters', 'avg_time']
    cluster_avg = cluster_avg.sort_values('num_clusters')
    
    print(f"   Cluster averages: {cluster_avg.values}")
    
    # Set up bar chart for right panel
    x2 = np.arange(len(clusters))
    bar_width = 0.6
    
    # Create bars for cluster averages (RIGHT PANEL)
    cluster_colors = ['#2E7D32', '#43A047', '#66BB6A', '#81C784', '#A5D6A7']  # Green gradient
    colors = cluster_colors[:len(clusters)]
    times = []
    for cluster in clusters:
        cluster_data = cluster_avg[cluster_avg['num_clusters'] == cluster]
        if not cluster_data.empty:
            times.append(cluster_data['avg_time'].iloc[0])
        else:
            times.append(0)
    
    print(f"   Right panel times: {times}")
    bars = ax2.bar(x2, times, bar_width, color=colors, alpha=0.8)
    
    # Customize RIGHT PANEL
    ax2.set_title('Average Running Time by Cluster Count\n(Across All Algorithms)', 
                 fontweight='bold', fontsize=14)
    ax2.set_xlabel('Cluster Count', fontsize=12)
    ax2.set_ylabel('Average Running Time (seconds)', fontsize=12)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cluster_labels, fontsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    # Main title
    plt.suptitle('Algorithm Performance Analysis: Running Time vs Cluster Count', 
                fontweight='bold', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0, 0.95, 0.92], w_pad=3.0)
    
    print("✅ Running time multipanel plot created successfully")
    return fig

def create_summary_statistics_table(df):
    """
    Create summary statistics table for running time analysis
    """
    if df.empty:
        return None
    
    df_clean = df.dropna(subset=['num_clusters', 'algorithm', 'running_time']).copy()
    
    if df_clean.empty:
        return None
    
    # Create comparison table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate data for table
    table_data = []
    headers = ['Algorithm', 'Overall Average', '1 Cluster', '3 Clusters', '5 Clusters', 'Std Deviation', 'Performance Consistency']
    
    # Overall averages
    overall_avg = df_clean.groupby('algorithm')['running_time'].mean()
    
    algorithms = sorted(df_clean['algorithm'].unique())
    clusters = sorted(df_clean['num_clusters'].unique())
    
    for alg in algorithms:
        alg_data = df_clean[df_clean['algorithm'] == alg]
        
        # Overall average
        overall = alg_data['running_time'].mean()
        
        # Per cluster averages
        cluster_avgs = {}
        for cluster in clusters:
            cluster_data = alg_data[alg_data['num_clusters'] == cluster]
            if not cluster_data.empty:
                cluster_avgs[cluster] = cluster_data['running_time'].mean()
            else:
                cluster_avgs[cluster] = 0
        
        # Standard deviation across clusters
        cluster_values = [v for v in cluster_avgs.values() if v > 0]
        std_dev = np.std(cluster_values) if len(cluster_values) > 1 else 0
        
        # Performance consistency rating
        cv = (std_dev / overall * 100) if overall > 0 else 0  # Coefficient of variation
        if cv < 10:
            consistency = "Very Consistent"
        elif cv < 20:
            consistency = "Consistent"
        elif cv < 30:
            consistency = "Moderate"
        else:
            consistency = "Variable"
        
        # Build row
        row = [
            alg,
            f"{overall:.3f}",
            f"{cluster_avgs.get(1, 0):.3f}" if cluster_avgs.get(1, 0) > 0 else "N/A",
            f"{cluster_avgs.get(3, 0):.3f}" if cluster_avgs.get(3, 0) > 0 else "N/A",
            f"{cluster_avgs.get(5, 0):.3f}" if cluster_avgs.get(5, 0) > 0 else "N/A",
            f"{std_dev:.3f}",
            consistency
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.15, 0.12, 0.12, 0.12, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on performance
    for i in range(1, len(table_data) + 1):
        alg = table_data[i-1][0]
        
        # Color based on algorithm
        if alg == 'BRKGA':
            row_color = '#e3f2fd'
        elif alg == 'DE':
            row_color = '#ffebee'
        elif alg == 'GA':
            row_color = '#fce4ec'
        else:  # PSO
            row_color = '#e0f2f1'
        
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_color)
    
    plt.title('Running Time Performance Comparison: Cluster-Specific vs Overall Average', 
              fontsize=16, fontweight='bold', pad=30)
    
    return fig

def main():
    """Main function to run the running time cluster analysis"""
    base_path = os.getcwd()
    
    print("🔍 Running Time Analysis: Cluster Performance vs Cluster Average")
    print("=" * 70)
    print("Creating multipanel visualization:")
    print("  - Left Panel: Algorithm running time by cluster count")
    print("  - Right Panel: Average running time by cluster count (across all algorithms)")
    print("=" * 70)
    
    # Load data
    df = load_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure.")
        return
    
    print(f"\n📊 Total records loaded: {len(df)}")
    
    # Create visualizations
    print("\n📈 Creating running time cluster analysis plot...")
    fig1 = create_running_time_cluster_analysis(df)
    if fig1:
        plt.show()
    
    print("\n📋 Creating running time statistics table...")
    fig2 = create_summary_statistics_table(df)
    if fig2:
        plt.show()
    
    # Print quick insights
    df_clean = df.dropna(subset=['algorithm', 'running_time'])
    if not df_clean.empty:
        print("\n" + "=" * 70)
        print("📊 QUICK INSIGHTS - RUNNING TIME")
        print("=" * 70)
        
        # Overall ranking by running time
        overall_ranking = df_clean.groupby('algorithm')['running_time'].mean().sort_values()
        print("⚡ Algorithm Ranking by Speed (averaged across all clusters):")
        for i, (alg, avg_time) in enumerate(overall_ranking.items(), 1):
            print(f"  {i}. {alg}: {avg_time:.3f}s average time")
        
        # Cluster impact analysis for running time
        cluster_ranking = df_clean.groupby('num_clusters')['running_time'].mean().sort_values()
        print(f"\n📈 Cluster Count Impact on Running Time (averaged across all algorithms):")
        for cluster, avg_time in cluster_ranking.items():
            print(f"  {int(cluster)} cluster{'s' if cluster != 1 else ''}: {avg_time:.3f}s average time")
    
    # Save option
    save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
    
    if save_plots == 'y':
        if fig1:
            fig1.savefig('running_time_cluster_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved running_time_cluster_analysis.png")
        
        if fig2:
            fig2.savefig('running_time_statistics_table.png', dpi=300, bbox_inches='tight')
            print("✅ Saved running_time_statistics_table.png")
        
        print("\n🎉 Running time cluster analysis complete!")

if __name__ == "__main__":
    main()