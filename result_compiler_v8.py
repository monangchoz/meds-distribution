import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
from matplotlib.ticker import FuncFormatter

def format_large_number(x, pos):
    """Format large numbers for better readability"""
    if x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'

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

def create_data_type_cost_comparison(df):
    """
    Create multipanel plot comparing average total cost between generated and historical data
    """
    if df.empty:
        print("No data available for analysis")
        return None
    
    # Clean data - remove missing values
    df_clean = df.dropna(subset=['data_type', 'algorithm', 'total_cost']).copy()
    
    if df_clean.empty:
        print("No clean data available")
        return None
    
    # Filter for generated and historical data
    generated_data = df_clean[df_clean['data_type'].str.lower() == 'generated'].copy()
    historical_data = df_clean[df_clean['data_type'].str.lower() == 'historical'].copy()
    
    if generated_data.empty or historical_data.empty:
        print("⚠️ Missing generated or historical data")
        return None
    
    print(f"📊 Creating data type cost comparison:")
    print(f"  - Generated data records: {len(generated_data)}")
    print(f"  - Historical data records: {len(historical_data)}")
    print(f"  - Algorithms: {sorted(df_clean['algorithm'].unique())}")
    
    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get unique algorithms
    algorithms = sorted(df_clean['algorithm'].unique())
    
    # Set colors to match previous plots
    algorithm_colors = {
        'BRKGA': '#1f77b4',  # Blue
        'DE': '#d62728',     # Red  
        'GA': '#e377c2',     # Pink
        'PSO': '#17becf'     # Cyan
    }
    
    # === LEFT PANEL: GENERATED DATA AVERAGE COST BY ALGORITHM ===
    print("📊 Creating left panel: Generated data average cost")
    
    # Calculate average cost per algorithm for generated data
    generated_avg = generated_data.groupby('algorithm').agg({
        'total_cost': 'mean'
    }).reset_index()
    
    generated_avg.columns = ['algorithm', 'avg_cost']
    generated_avg = generated_avg.sort_values('algorithm')
    
    # Set up bar chart for left panel
    x1 = np.arange(len(algorithms))
    bar_width = 0.6
    
    # Create bars for generated data (LEFT PANEL)
    colors1 = [algorithm_colors.get(alg, f'C{i}') for i, alg in enumerate(algorithms)]
    costs1 = []
    for alg in algorithms:
        alg_data = generated_avg[generated_avg['algorithm'] == alg]
        if not alg_data.empty:
            costs1.append(alg_data['avg_cost'].iloc[0])
        else:
            costs1.append(0)
    
    bars1 = ax1.bar(x1, costs1, bar_width, color=colors1, alpha=0.8)
    
    # Customize LEFT PANEL
    ax1.set_title('Average Total Cost - Generated Data', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Average Total Cost', fontsize=12)
    
    ax1.set_xticks(x1)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    print(f"   Generated data costs: {costs1}")
    
    # === RIGHT PANEL: HISTORICAL DATA AVERAGE COST BY ALGORITHM ===
    print("📊 Creating right panel: Historical data average cost")
    
    # Calculate average cost per algorithm for historical data
    historical_avg = historical_data.groupby('algorithm').agg({
        'total_cost': 'mean'
    }).reset_index()
    
    historical_avg.columns = ['algorithm', 'avg_cost']
    historical_avg = historical_avg.sort_values('algorithm')
    
    # Set up bar chart for right panel
    x2 = np.arange(len(algorithms))
    
    # Create bars for historical data (RIGHT PANEL)
    colors2 = [algorithm_colors.get(alg, f'C{i}') for i, alg in enumerate(algorithms)]
    costs2 = []
    for alg in algorithms:
        alg_data = historical_avg[historical_avg['algorithm'] == alg]
        if not alg_data.empty:
            costs2.append(alg_data['avg_cost'].iloc[0])
        else:
            costs2.append(0)
    
    bars2 = ax2.bar(x2, costs2, bar_width, color=colors2, alpha=0.8)
    
    # Customize RIGHT PANEL
    ax2.set_title('Average Total Cost - Historical Data', 
                 fontweight='bold', fontsize=14)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Average Total Cost', fontsize=12)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(algorithms, fontsize=11)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    ax2.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    print(f"   Historical data costs: {costs2}")
    
    # Make y-axis scales consistent for better comparison
    max_cost = max(max(costs1), max(costs2))
    ax1.set_ylim(0, max_cost * 1.1)
    ax2.set_ylim(0, max_cost * 1.1)
    
    # Main title
    plt.suptitle('Average Total Cost Comparison: Generated vs Historical Data', 
                fontweight='bold', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0, 0.95, 0.92], w_pad=3.0)
    
    print("✅ Data type cost comparison plot created successfully")
    return fig

def create_cost_difference_analysis(df):
    """
    Create analysis of cost differences between generated and historical data
    """
    if df.empty:
        return None
    
    df_clean = df.dropna(subset=['data_type', 'algorithm', 'total_cost']).copy()
    
    if df_clean.empty:
        return None
    
    # Calculate average costs by algorithm and data type
    cost_comparison = df_clean.groupby(['algorithm', 'data_type']).agg({
        'total_cost': 'mean'
    }).reset_index()
    
    # Pivot to have generated and historical as columns
    cost_pivot = cost_comparison.pivot(index='algorithm', columns='data_type', values='total_cost')
    
    # Calculate percentage difference: (Generated - Historical) / Historical * 100
    if 'generated' in cost_pivot.columns and 'historical' in cost_pivot.columns:
        cost_pivot['difference_pct'] = ((cost_pivot['generated'] - cost_pivot['historical']) / 
                                      cost_pivot['historical'] * 100)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for table
    table_data = []
    headers = ['Algorithm', 'Generated Avg', 'Historical Avg', 'Difference', 'Difference %']
    
    algorithms = sorted(cost_pivot.index)
    
    for alg in algorithms:
        if alg in cost_pivot.index:
            generated_cost = cost_pivot.loc[alg, 'generated'] if 'generated' in cost_pivot.columns else 0
            historical_cost = cost_pivot.loc[alg, 'historical'] if 'historical' in cost_pivot.columns else 0
            difference = generated_cost - historical_cost
            difference_pct = cost_pivot.loc[alg, 'difference_pct'] if 'difference_pct' in cost_pivot.columns else 0
            
            row = [
                alg,
                f"{generated_cost:,.0f}",
                f"{historical_cost:,.0f}",
                f"{difference:,.0f}",
                f"{difference_pct:.1f}%"
            ]
            table_data.append(row)
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.20, 0.20, 0.20, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on performance difference
    for i in range(1, len(table_data) + 1):
        difference_pct = float(table_data[i-1][4].replace('%', ''))
        
        if difference_pct > 50:
            row_color = '#ffcdd2'  # Light red for high difference
        elif difference_pct > 0:
            row_color = '#fff3e0'  # Light orange for positive difference
        else:
            row_color = '#e8f5e8'  # Light green for negative difference
        
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_color)
    
    plt.title('Cost Comparison Analysis: Generated vs Historical Data', 
              fontsize=16, fontweight='bold', pad=30)
    
    return fig

def main():
    """Main function to run the data type cost comparison analysis"""
    base_path = os.getcwd()
    
    print("🔍 Data Type Cost Comparison: Generated vs Historical")
    print("=" * 65)
    print("Creating multipanel visualization:")
    print("  - Left Panel: Average total cost for Generated data by algorithm")
    print("  - Right Panel: Average total cost for Historical data by algorithm")
    print("=" * 65)
    
    # Load data
    df = load_data(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure.")
        return
    
    print(f"\n📊 Total records loaded: {len(df)}")
    
    # Create visualizations
    print("\n📈 Creating data type cost comparison plot...")
    fig1 = create_data_type_cost_comparison(df)
    if fig1:
        plt.show()
    
    print("\n📋 Creating cost difference analysis table...")
    fig2 = create_cost_difference_analysis(df)
    if fig2:
        plt.show()
    
    # Print quick insights
    df_clean = df.dropna(subset=['algorithm', 'total_cost', 'data_type'])
    if not df_clean.empty:
        print("\n" + "=" * 65)
        print("📊 QUICK INSIGHTS - COST COMPARISON")
        print("=" * 65)
        
        # Overall comparison
        generated_overall = df_clean[df_clean['data_type'].str.lower() == 'generated']['total_cost'].mean()
        historical_overall = df_clean[df_clean['data_type'].str.lower() == 'historical']['total_cost'].mean()
        
        print(f"📈 Overall Average Costs:")
        print(f"  Generated Data: {generated_overall:,.0f}")
        print(f"  Historical Data: {historical_overall:,.0f}")
        print(f"  Difference: {generated_overall - historical_overall:,.0f} ({((generated_overall - historical_overall) / historical_overall * 100):.1f}%)")
        
        # Best algorithm for each data type
        generated_best = df_clean[df_clean['data_type'].str.lower() == 'generated'].groupby('algorithm')['total_cost'].mean().idxmin()
        historical_best = df_clean[df_clean['data_type'].str.lower() == 'historical'].groupby('algorithm')['total_cost'].mean().idxmin()
        
        print(f"\n🏆 Best Algorithm by Data Type:")
        print(f"  Generated Data: {generated_best}")
        print(f"  Historical Data: {historical_best}")
    
    # Save option
    save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
    
    if save_plots == 'y':
        if fig1:
            fig1.savefig('data_type_cost_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Saved data_type_cost_comparison.png")
        
        if fig2:
            fig2.savefig('cost_difference_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved cost_difference_analysis.png")
        
        print("\n🎉 Data type cost comparison analysis complete!")

if __name__ == "__main__":
    main()