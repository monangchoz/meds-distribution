import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from matplotlib.ticker import FuncFormatter

def extract_cluster_info(filename):
    """
    Extract cluster information from filename
    Returns: number of clusters (1, 3, or 5) or None if not found
    """
    filename_upper = filename.upper()
    if 'NCL_1' in filename_upper:
        return 1
    elif 'NCL_3' in filename_upper:
        return 3
    elif 'NCL_5' in filename_upper:
        return 5
    return None

def extract_data_type(filename):
    """
    Extract data type (historical or generated) from filename
    Returns: 'historical', 'generated', or None if not found
    """
    filename_upper = filename.upper()
    if 'HISTORICAL' in filename_upper:
        return 'historical'
    elif 'GENERATED' in filename_upper:
        return 'generated'
    return None

def load_and_process_csv_files(base_path):
    """
    Load all CSV files from algorithm folders and organize by multiple dimensions:
    - Location (JK2, MKS, SBY)
    - Algorithm (AVNS, GA, DE, BRKGA, PSO)
    - Clusters (1, 3, 5)
    - Data Type (historical, generated)
    """
    # Multi-dimensional dictionary structure
    data_by_location = {
        'JK2': {},
        'MKS': {},
        'SBY': {}
    }
    
    # Also create separate structures for cluster and data type analysis
    data_by_cluster = {1: {}, 3: {}, 5: {}}
    data_by_datatype = {'historical': {}, 'generated': {}}
    
    # Define algorithm folders and their display names
    algorithm_folders = {
        'avns': 'AVNS',
        'ga': 'GA', 
        'de': 'DE',
        'brkga': 'BRKGA',
        'pso': 'PSO'
    }
    
    results_path = os.path.join(base_path, 'results')
    
    if not os.path.exists(results_path):
        print(f"Results folder not found at: {results_path}")
        return data_by_location, data_by_cluster, data_by_datatype
    
    # Process each algorithm folder
    for folder_name, algorithm_name in algorithm_folders.items():
        algorithm_path = os.path.join(results_path, folder_name)
        
        if not os.path.exists(algorithm_path):
            print(f"Algorithm folder not found: {algorithm_path}")
            continue
            
        # Get all CSV files in this algorithm folder
        csv_files = glob.glob(os.path.join(algorithm_path, "*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            filename_upper = filename.upper()
            
            # Extract all dimensions
            location = None
            if 'JK2' in filename_upper:
                location = 'JK2'
            elif 'MKS' in filename_upper:
                location = 'MKS'
            elif 'SBY' in filename_upper:
                location = 'SBY'
            
            cluster_count = extract_cluster_info(filename)
            data_type = extract_data_type(filename)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, header=None, names=['total_cost', 'running_time'])
                
                # Store in location-based structure (existing functionality)
                if location:
                    if algorithm_name not in data_by_location[location]:
                        data_by_location[location][algorithm_name] = {'total_cost': [], 'running_time': []}
                    data_by_location[location][algorithm_name]['total_cost'].extend(df['total_cost'].tolist())
                    data_by_location[location][algorithm_name]['running_time'].extend(df['running_time'].tolist())
                
                # Store in cluster-based structure
                if cluster_count is not None:
                    if algorithm_name not in data_by_cluster[cluster_count]:
                        data_by_cluster[cluster_count][algorithm_name] = {'total_cost': [], 'running_time': []}
                    data_by_cluster[cluster_count][algorithm_name]['total_cost'].extend(df['total_cost'].tolist())
                    data_by_cluster[cluster_count][algorithm_name]['running_time'].extend(df['running_time'].tolist())
                
                # Store in data type-based structure
                if data_type:
                    if algorithm_name not in data_by_datatype[data_type]:
                        data_by_datatype[data_type][algorithm_name] = {'total_cost': [], 'running_time': []}
                    data_by_datatype[data_type][algorithm_name]['total_cost'].extend(df['total_cost'].tolist())
                    data_by_datatype[data_type][algorithm_name]['running_time'].extend(df['running_time'].tolist())
                
                # Enhanced logging
                cluster_info = f", Clusters: {cluster_count}" if cluster_count else ""
                datatype_info = f", Data: {data_type}" if data_type else ""
                print(f"Processed {filename} -> Location: {location}, Algorithm: {algorithm_name}{cluster_info}{datatype_info}, Rows: {len(df)}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return data_by_location, data_by_cluster, data_by_datatype

# Custom formatter for large numbers
def format_large_number(x, pos):
    if x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'

def create_cost_visualization(location, algorithms_data, location_name):
    """
    Create visualization focused on COST analysis for one location - box plot only
    """
    # Set up the figure for a single plot (box plot only)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'{location_name} - Total Cost Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    algorithms = list(algorithms_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    if not algorithms:
        fig.text(0.5, 0.5, f'No data available for {location_name}', 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Total Cost Box Plot with simpler labels
    cost_data = []
    cost_labels = []
    
    for alg in algorithms:
        if algorithms_data[alg]['total_cost']:
            cost_data.append(algorithms_data[alg]['total_cost'])
            # Simplified labels without numbers
            cost_labels.append(f"{alg}")
    
    if cost_data:
        bp = ax.boxplot(cost_data, labels=cost_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Total Cost Distribution by Algorithm', fontweight='bold', fontsize=14)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        
        # Format y-axis for large numbers
        formatter = FuncFormatter(format_large_number)
        ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def create_time_visualization(location, algorithms_data, location_name):
    """
    Create visualization focused on RUNNING TIME analysis for one location - box plot only
    """
    # Set up the figure for a single plot (box plot only)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'{location_name} - Running Time Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    algorithms = list(algorithms_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    if not algorithms:
        fig.text(0.5, 0.5, f'No data available for {location_name}', 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Running Time Box Plot with simpler labels
    time_data = []
    time_labels = []
    
    for alg in algorithms:
        if algorithms_data[alg]['running_time']:
            time_data.append(algorithms_data[alg]['running_time'])
            # Simplified labels
            time_labels.append(f"{alg}")
    
    if time_data:
        bp = ax.boxplot(time_data, labels=time_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Running Time Distribution by Algorithm', fontweight='bold', fontsize=14)
        ax.set_ylabel('Running Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def create_cluster_analysis_visualization(cluster_data, analysis_type):
    """
    Create visualization comparing algorithms across different cluster counts
    analysis_type: 'cost' or 'time'
    """
    clusters = [1, 3, 5]
    cluster_labels = ['1 Cluster', '3 Clusters', '5 Clusters']
    
    # Get all algorithms across all clusters
    all_algorithms = set()
    for cluster_count in clusters:
        if cluster_count in cluster_data:
            all_algorithms.update(cluster_data[cluster_count].keys())
    all_algorithms = sorted(list(all_algorithms))
    
    if not all_algorithms:
        print(f"No data available for cluster analysis")
        return None
    
    # Set up the figure with more height for the lower plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [1, 1.2]})
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    
    metric = 'total_cost' if analysis_type == 'cost' else 'running_time'
    metric_label = 'Total Cost' if analysis_type == 'cost' else 'Running Time'
    y_label = metric_label if analysis_type == 'cost' else f'{metric_label} (seconds)'
    
    fig.suptitle(f'{metric_label} Analysis by Cluster Count', fontsize=18, fontweight='bold')
    
    # Get plots per cluster count - 1st row
    cluster_axes = {
        1: ax1,
        3: ax2,
        5: ax3
    }
    
    # Format y-axis for large numbers if we're dealing with cost
    formatter = FuncFormatter(format_large_number) if analysis_type == 'cost' else None
    
    # 1. Box plots for each cluster count - with simplified labels
    for cluster_count, label in zip(clusters, cluster_labels):
        if cluster_count in cluster_data and cluster_data[cluster_count]:
            plot_data = []
            plot_labels = []
            
            for alg in all_algorithms:
                if alg in cluster_data[cluster_count] and cluster_data[cluster_count][alg][metric]:
                    data_values = cluster_data[cluster_count][alg][metric]
                    plot_data.append(data_values)
                    # Simplified labels with just algorithm name
                    plot_labels.append(f"{alg}")
            
            if plot_data:
                ax = cluster_axes[cluster_count]
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_title(f'{label}', fontweight='bold', fontsize=14)
                ax.set_ylabel(y_label, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                
                # Apply formatter for cost data
                if formatter:
                    ax.yaxis.set_major_formatter(formatter)
    
    # 2. IMPROVED CHART: Replace errorbar with grouped bar chart for better readability
    width = 0.15  # width of bars
    x = np.arange(len(clusters))  # cluster counts on x-axis
    
    # Use a different colormap for better distinction
    alg_colors = plt.cm.tab10(np.linspace(0, 1, len(all_algorithms)))
    
    for i, alg in enumerate(all_algorithms):
        means = []
        for cluster_count in clusters:
            if (cluster_count in cluster_data and 
                alg in cluster_data[cluster_count] and 
                cluster_data[cluster_count][alg][metric]):
                
                data_values = cluster_data[cluster_count][alg][metric]
                means.append(np.mean(data_values))
            else:
                means.append(0)
                
        # Position bars side by side with an offset based on algorithm index
        offset = (i - len(all_algorithms)/2 + 0.5) * width
        ax4.bar(x + offset, means, width, label=alg, color=alg_colors[i], alpha=0.85)
    
    ax4.set_title('Algorithm Performance Across Cluster Counts', fontweight='bold', fontsize=14)
    ax4.set_ylabel(y_label, fontsize=12)
    ax4.set_xlabel('Cluster Count', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cluster_labels)
    # Place legend outside the plot to avoid overlap
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(5, len(all_algorithms)), fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Apply formatter for cost data
    if formatter:
        ax4.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust for suptitle and legend
    plt.subplots_adjust(hspace=0.4)  # Add more space between rows
    return fig

def create_historical_data_visualization(datatype_data, analysis_type):
    """
    Create visualization for algorithms on historical data only - box plot only
    analysis_type: 'cost' or 'time'
    """
    # Get all algorithms for historical data
    all_algorithms = []
    if 'historical' in datatype_data:
        all_algorithms = sorted(list(datatype_data['historical'].keys()))
    
    if not all_algorithms:
        print("No data available for historical data analysis")
        return None
    
    # Set up the figure for a single plot (box plot only)
    fig, ax = plt.subplots(figsize=(12, 7))
    metric = 'total_cost' if analysis_type == 'cost' else 'running_time'
    metric_label = 'Total Cost' if analysis_type == 'cost' else 'Running Time'
    y_label = metric_label if analysis_type == 'cost' else f'{metric_label} (seconds)'
    
    fig.suptitle(f'{metric_label} Analysis - Historical Data', fontsize=18, fontweight='bold')
    
    # Format y-axis for large numbers if we're dealing with cost
    formatter = FuncFormatter(format_large_number) if analysis_type == 'cost' else None
    
    # Box plot for historical data
    plot_data = []
    plot_labels = []
    
    for alg in all_algorithms:
        if alg in datatype_data['historical'] and datatype_data['historical'][alg][metric]:
            data_values = datatype_data['historical'][alg][metric]
            plot_data.append(data_values)
            # Simple algorithm name without values in labels
            plot_labels.append(f"{alg}")
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Distribution by Algorithm', fontweight='bold', fontsize=14)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Apply formatter for cost data
        if formatter:
            ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def create_generated_data_visualization(datatype_data, analysis_type):
    """
    Create visualization for algorithms on generated data only - box plot only
    analysis_type: 'cost' or 'time'
    """
    # Get all algorithms for generated data
    all_algorithms = []
    if 'generated' in datatype_data:
        all_algorithms = sorted(list(datatype_data['generated'].keys()))
    
    if not all_algorithms:
        print("No data available for generated data analysis")
        return None
    
    # Set up the figure for a single plot (box plot only)
    fig, ax = plt.subplots(figsize=(12, 7))
    metric = 'total_cost' if analysis_type == 'cost' else 'running_time'
    metric_label = 'Total Cost' if analysis_type == 'cost' else 'Running Time'
    y_label = metric_label if analysis_type == 'cost' else f'{metric_label} (seconds)'
    
    fig.suptitle(f'{metric_label} Analysis - Generated Data', fontsize=18, fontweight='bold')
    
    # Format y-axis for large numbers if we're dealing with cost
    formatter = FuncFormatter(format_large_number) if analysis_type == 'cost' else None
    
    # Box plot for generated data
    plot_data = []
    plot_labels = []
    
    for alg in all_algorithms:
        if alg in datatype_data['generated'] and datatype_data['generated'][alg][metric]:
            data_values = datatype_data['generated'][alg][metric]
            plot_data.append(data_values)
            # Simple algorithm name without values in labels
            plot_labels.append(f"{alg}")
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Distribution by Algorithm', fontweight='bold', fontsize=14)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Apply formatter for cost data
        if formatter:
            ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def create_datatype_comparison_visualization(datatype_data, analysis_type):
    """
    Create visualization comparing algorithms between historical and generated data
    analysis_type: 'cost' or 'time'
    """
    data_types = ['historical', 'generated']
    data_type_labels = ['Historical', 'Generated']
    
    # Get all algorithms across both data types
    all_algorithms = set()
    for data_type in data_types:
        if data_type in datatype_data:
            all_algorithms.update(datatype_data[data_type].keys())
    all_algorithms = sorted(list(all_algorithms))
    
    if not all_algorithms:
        print(f"No data available for data type comparison")
        return None
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    metric = 'total_cost' if analysis_type == 'cost' else 'running_time'
    metric_label = 'Total Cost' if analysis_type == 'cost' else 'Running Time'
    y_label = metric_label if analysis_type == 'cost' else f'{metric_label} (seconds)'
    
    fig.suptitle(f'{metric_label} Comparison: Historical vs Generated Data', fontsize=18, fontweight='bold')
    
    # Format y-axis for large numbers if we're dealing with cost
    formatter = FuncFormatter(format_large_number) if analysis_type == 'cost' else None
    
    # 1. BOX PLOT COMPARISON (replacing grouped bar chart)
    # Create positions for box plots
    positions = []
    labels = []
    box_data = []
    box_colors = []
    
    # Colors for historical and generated data
    hist_color = '#3498db'  # blue
    gen_color = '#e74c3c'   # red
    
    # Create data for box plot
    for i, alg in enumerate(all_algorithms):
        # Historical data
        if ('historical' in datatype_data and 
            alg in datatype_data['historical'] and 
            datatype_data['historical'][alg][metric]):
            
            # Position calculations for grouped boxplots
            pos = i * 3  # Each algorithm gets 3 units of space
            positions.append(pos)
            labels.append(f"{alg}\nHistorical")
            box_data.append(datatype_data['historical'][alg][metric])
            box_colors.append(hist_color)
        
        # Generated data
        if ('generated' in datatype_data and 
            alg in datatype_data['generated'] and 
            datatype_data['generated'][alg][metric]):
            
            # Position right next to historical box
            pos = i * 3 + 1  # +1 to position next to historical
            positions.append(pos)
            labels.append(f"{alg}\nGenerated")
            box_data.append(datatype_data['generated'][alg][metric])
            box_colors.append(gen_color)
    
    # Create box plot
    if box_data:
        bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)
        
        # Customize box colors
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add a legend
        hist_patch = plt.Rectangle((0, 0), 1, 1, fc=hist_color, alpha=0.7)
        gen_patch = plt.Rectangle((0, 0), 1, 1, fc=gen_color, alpha=0.7)
        ax1.legend([hist_patch, gen_patch], ['Historical', 'Generated'], loc='upper right')
        
        # Set labels and title
        ax1.set_title('Distribution Comparison by Algorithm', fontweight='bold', fontsize=14)
        ax1.set_ylabel(y_label, fontsize=12)
        
        # Set x-axis ticks at the middle of each algorithm's boxes
        algorithm_positions = [i * 3 + 0.5 for i in range(len(all_algorithms))]
        ax1.set_xticks(algorithm_positions)
        ax1.set_xticklabels(all_algorithms, rotation=45, ha='right')
        
        # Add grid
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Apply formatter for cost data
        if formatter:
            ax1.yaxis.set_major_formatter(formatter)
    
    # 2. Difference analysis (Generated - Historical) - KEEPING THIS THE SAME
    differences = []
    difference_labels = []
    historical_means = []
    generated_means = []
    
    # Calculate means for historical and generated data
    for alg in all_algorithms:
        # Historical data
        if ('historical' in datatype_data and 
            alg in datatype_data['historical'] and 
            datatype_data['historical'][alg][metric]):
            hist_data = datatype_data['historical'][alg][metric]
            historical_means.append(np.mean(hist_data))
        else:
            historical_means.append(0)
        
        # Generated data
        if ('generated' in datatype_data and 
            alg in datatype_data['generated'] and 
            datatype_data['generated'][alg][metric]):
            gen_data = datatype_data['generated'][alg][metric]
            generated_means.append(np.mean(gen_data))
        else:
            generated_means.append(0)
    
    # Calculate percentage differences
    for i, alg in enumerate(all_algorithms):
        if historical_means[i] > 0 and generated_means[i] > 0:
            diff = generated_means[i] - historical_means[i]
            diff_percent = (diff / historical_means[i]) * 100
            differences.append(diff_percent)
            difference_labels.append(alg)
    
    if differences:
        colors = ['red' if x > 0 else 'green' for x in differences]
        bars = ax2.bar(range(len(differences)), differences, color=colors, alpha=0.7)
        ax2.set_title('Performance Difference (Generated - Historical) %', fontweight='bold', fontsize=14)
        ax2.set_ylabel(f'{metric_label} Difference (%)', fontsize=12)
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_xticks(range(len(difference_labels)))
        ax2.set_xticklabels(difference_labels, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3)
        
        # Keep percentage values on bars
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            # Position label based on bar height
            if height > 0:
                v_align = 'bottom'
                y_pos = height + 1
            else:
                v_align = 'top'
                y_pos = height - 3
                
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{diff:.1f}%', ha='center', va=v_align, fontsize=9,
                    fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def create_summary_comparison(data_by_location):
    """
    Create a summary comparison across all locations
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14), 
                                                gridspec_kw={'height_ratios': [1, 0.9]})
    fig.suptitle('Cross-Location Algorithm Comparison Summary', fontsize=18, fontweight='bold')
    
    # Collect all algorithms across locations
    all_algorithms = set()
    for location_data in data_by_location.values():
        all_algorithms.update(location_data.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    locations = ['JK2', 'MKS', 'SBY']
    location_colors = {'JK2': '#3498db', 'MKS': '#e74c3c', 'SBY': '#2ecc71'}
    
    # Format for large numbers
    formatter = FuncFormatter(format_large_number)
    
    # 1. Average Cost by Location and Algorithm (Grouped Bar Chart)
    cost_means = {}
    for location in locations:
        cost_means[location] = []
        for alg in all_algorithms:
            if alg in data_by_location[location] and data_by_location[location][alg]['total_cost']:
                cost_means[location].append(np.mean(data_by_location[location][alg]['total_cost']))
            else:
                cost_means[location].append(0)
    
    x = np.arange(len(all_algorithms))
    width = 0.25
    
    for i, location in enumerate(locations):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, cost_means[location], width, label=location, 
                      color=location_colors[location], alpha=0.85)
    
    ax1.set_title('Average Total Cost by Algorithm and Location', fontweight='bold')
    ax1.set_ylabel('Average Total Cost')
    ax1.set_xlabel('Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_algorithms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(formatter)
    
    # 2. Average Time by Location and Algorithm (Grouped Bar Chart)
    time_means = {}
    for location in locations:
        time_means[location] = []
        for alg in all_algorithms:
            if alg in data_by_location[location] and data_by_location[location][alg]['running_time']:
                time_means[location].append(np.mean(data_by_location[location][alg]['running_time']))
            else:
                time_means[location].append(0)
    
    for i, location in enumerate(locations):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, time_means[location], width, label=location, 
                      color=location_colors[location], alpha=0.85)
    
    ax2.set_title('Average Running Time by Algorithm and Location', fontweight='bold')
    ax2.set_ylabel('Average Running Time (seconds)')
    ax2.set_xlabel('Algorithm')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_algorithms, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost Distribution Heatmap
    cost_matrix = np.array([cost_means[loc] for loc in locations])
    im1 = ax3.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_title('Cost Performance Heatmap', fontweight='bold')
    ax3.set_yticks(range(len(locations)))
    ax3.set_yticklabels(locations)
    ax3.set_xticks(range(len(all_algorithms)))
    ax3.set_xticklabels(all_algorithms, rotation=45, ha='right')
    
    # No value annotations as requested
    
    cbar1 = plt.colorbar(im1, ax=ax3, format=formatter)
    cbar1.set_label('Average Total Cost')
    
    # 4. Time Distribution Heatmap
    time_matrix = np.array([time_means[loc] for loc in locations])
    im2 = ax4.imshow(time_matrix, cmap='YlGnBu', aspect='auto')
    ax4.set_title('Time Performance Heatmap', fontweight='bold')
    ax4.set_yticks(range(len(locations)))
    ax4.set_yticklabels(locations)
    ax4.set_xticks(range(len(all_algorithms)))
    ax4.set_xticklabels(all_algorithms, rotation=45, ha='right')
    
    # No value annotations as requested
    
    cbar2 = plt.colorbar(im2, ax=ax4)
    cbar2.set_label('Average Running Time (seconds)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.subplots_adjust(hspace=0.25)  # Add space between rows
    
    return fig

def print_detailed_statistics(data_by_location):
    """
    Print comprehensive statistics for each location and algorithm
    """
    print("\n" + "="*100)
    print("DETAILED ALGORITHM PERFORMANCE STATISTICS")
    print("="*100)
    
    for location in ['JK2', 'MKS', 'SBY']:
        if data_by_location[location]:
            print(f"\n{'='*20} {location} LOCATION {'='*20}")
            
            for algorithm, data in data_by_location[location].items():
                if data['total_cost']:
                    print(f"\n{algorithm}:")
                    print("-" * 60)
                    
                    # Total Cost Statistics
                    costs = data['total_cost']
                    print(f"Total Cost (n={len(costs)}):")
                    print(f"  Mean: {np.mean(costs):,.2f}")
                    print(f"  Median: {np.median(costs):,.2f}")
                    print(f"  Std Dev: {np.std(costs):,.2f}")
                    print(f"  Min: {np.min(costs):,.2f}")
                    print(f"  Max: {np.max(costs):,.2f}")
                    
                    # Running Time Statistics
                    times = data['running_time']
                    print(f"Running Time (n={len(times)}):")
                    print(f"  Mean: {np.mean(times):.6f}")
                    print(f"  Median: {np.median(times):.6f}")
                    print(f"  Std Dev: {np.std(times):.6f}")
                    print(f"  Min: {np.min(times):.6f}")
                    print(f"  Max: {np.max(times):.6f}")
        else:
            print(f"\n{location} Location: No data available")

def main():
    # Base path is the current directory where the script is located
    base_path = os.getcwd()  # Current working directory
    
    print(f"Looking for CSV files in: {base_path}/results/")
    print("Expected folder structure:")
    print("  - results/avns/")
    print("  - results/ga/") 
    print("  - results/de/")
    print("  - results/brkga/")
    print("  - results/pso/")
    print("\nExpected filename patterns:")
    print("  - Location codes: JK2, MKS, SBY")
    print("  - Cluster codes: ncl_1, ncl_3, ncl_5")
    print("  - Data type codes: historical, generated")
    print("-" * 50)
    
    data_by_location, data_by_cluster, data_by_datatype = load_and_process_csv_files(base_path)
    
    # Print detailed statistics
    print_detailed_statistics(data_by_location)
    
    all_figures = []
    
    # ===== LOCATION-BASED ANALYSIS =====
    print("\n" + "="*60)
    print("CREATING LOCATION-BASED VISUALIZATIONS")
    print("="*60)
    
    locations = ['JK2', 'MKS', 'SBY']
    location_names = ['Jakarta 2', 'Makassar', 'Surabaya']
    
    for location, name in zip(locations, location_names):
        if data_by_location[location]:
            print(f"\nCreating COST visualization for {location}...")
            cost_fig = create_cost_visualization(location, data_by_location[location], name)
            all_figures.append((cost_fig, f"{location}_cost_analysis"))
            plt.show()
            
            print(f"Creating TIME visualization for {location}...")
            time_fig = create_time_visualization(location, data_by_location[location], name)
            all_figures.append((time_fig, f"{location}_time_analysis"))
            plt.show()
        else:
            print(f"No data found for {location}")
    
    # ===== CLUSTER-BASED ANALYSIS =====
    print("\n" + "="*60)
    print("CREATING CLUSTER-BASED VISUALIZATIONS")
    print("="*60)
    
    if any(data_by_cluster.values()):
        print("Creating cluster-based COST analysis...")
        cluster_cost_fig = create_cluster_analysis_visualization(data_by_cluster, 'cost')
        if cluster_cost_fig:
            all_figures.append((cluster_cost_fig, "cluster_cost_analysis"))
            plt.show()
        
        print("Creating cluster-based TIME analysis...")
        cluster_time_fig = create_cluster_analysis_visualization(data_by_cluster, 'time')
        if cluster_time_fig:
            all_figures.append((cluster_time_fig, "cluster_time_analysis"))
            plt.show()
    else:
        print("No cluster data found. Make sure filenames contain 'ncl_1', 'ncl_3', or 'ncl_5'")
    
    # ===== DATA TYPE-BASED ANALYSIS - SEPARATED =====
    print("\n" + "="*60)
    print("CREATING DATA TYPE-BASED VISUALIZATIONS (SEPARATED)")
    print("="*60)
    
    if any(data_by_datatype.values()):
        # Historical Data visualizations
        if 'historical' in data_by_datatype and data_by_datatype['historical']:
            print("Creating HISTORICAL data COST analysis...")
            hist_cost_fig = create_historical_data_visualization(data_by_datatype, 'cost')
            if hist_cost_fig:
                all_figures.append((hist_cost_fig, "historical_cost_analysis"))
                plt.show()
            
            print("Creating HISTORICAL data TIME analysis...")
            hist_time_fig = create_historical_data_visualization(data_by_datatype, 'time')
            if hist_time_fig:
                all_figures.append((hist_time_fig, "historical_time_analysis"))
                plt.show()
        else:
            print("No historical data found.")
        
        # Generated Data visualizations
        if 'generated' in data_by_datatype and data_by_datatype['generated']:
            print("Creating GENERATED data COST analysis...")
            gen_cost_fig = create_generated_data_visualization(data_by_datatype, 'cost')
            if gen_cost_fig:
                all_figures.append((gen_cost_fig, "generated_cost_analysis"))
                plt.show()
            
            print("Creating GENERATED data TIME analysis...")
            gen_time_fig = create_generated_data_visualization(data_by_datatype, 'time')
            if gen_time_fig:
                all_figures.append((gen_time_fig, "generated_time_analysis"))
                plt.show()
        else:
            print("No generated data found.")
        
        # Comparison visualization
        print("Creating Historical vs. Generated comparison COST analysis...")
        comparison_cost_fig = create_datatype_comparison_visualization(data_by_datatype, 'cost')
        if comparison_cost_fig:
            all_figures.append((comparison_cost_fig, "datatype_comparison_cost"))
            plt.show()
        
        print("Creating Historical vs. Generated comparison TIME analysis...")
        comparison_time_fig = create_datatype_comparison_visualization(data_by_datatype, 'time')
        if comparison_time_fig:
            all_figures.append((comparison_time_fig, "datatype_comparison_time"))
            plt.show()
    else:
        print("No data type data found. Make sure filenames contain 'historical' or 'generated'")
    
    # ===== SUMMARY COMPARISON =====
    print("\n" + "="*60)
    print("CREATING CROSS-LOCATION SUMMARY")
    print("="*60)
    
    summary_fig = create_summary_comparison(data_by_location)
    all_figures.append((summary_fig, "cross_location_summary"))
    plt.show()
    
    # ===== SAVE PLOTS =====
    save_plots = input(f"\nDo you want to save all {len(all_figures)} plots? (y/n): ").lower().strip()
    if save_plots == 'y':
        print("\nSaving plots...")
        for fig, name in all_figures:
            fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved {name}.png")
        
        print(f"\n🎉 Successfully saved {len(all_figures)} visualizations!")
        print("\nFiles created:")
        print("📍 Location-based analysis:")
        for location in ['JK2', 'MKS', 'SBY']:
            print(f"   - {location}_cost_analysis.png")
            print(f"   - {location}_time_analysis.png")
        print("🔢 Cluster-based analysis:")
        print("   - cluster_cost_analysis.png")
        print("   - cluster_time_analysis.png")
        print("📊 Data type-based analysis (SEPARATED):")
        print("   - historical_cost_analysis.png")
        print("   - historical_time_analysis.png")
        print("   - generated_cost_analysis.png")
        print("   - generated_time_analysis.png")
        print("   - datatype_comparison_cost.png")
        print("   - datatype_comparison_time.png")
        print("📈 Summary:")
        print("   - cross_location_summary.png")

if __name__ == "__main__":
    main()