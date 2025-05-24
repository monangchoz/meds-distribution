import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import wilcoxon, permutation_test
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_data(folder_path='.'):
    """
    Load CSV files organized by algorithm folders and with location identifiers in filenames.
    Each file contains rows with format: total_cost,running_time
    
    The function looks for algorithm folders (ga, de, brkga, pso, avns) and 
    location identifiers (JK2, MKS, SBY) in filenames.
    
    Args:
        folder_path: Path to the root folder containing algorithm folders
        
    Returns:
        Dictionary with data organized by place, metric, and algorithm
    """
    # Initialize data structure
    data = {
        'JK2': {'cost': {}, 'time': {}, 'instance_ids': set()},
        'MKS': {'cost': {}, 'time': {}, 'instance_ids': set()},
        'SBY': {'cost': {}, 'time': {}, 'instance_ids': set()}
    }
    
    # Define possible paths to search
    search_paths = [
        folder_path,  # Direct algorithm folders
        os.path.join(folder_path, 'results')  # Algorithm folders in 'results' directory
    ]
    
    # List of algorithms to look for
    algorithms = ['ga', 'de', 'brkga', 'pso', 'avns']
    # Also check for capitalized versions
    algorithms_caps = algorithms + [algo.upper() for algo in algorithms]
    
    found_files = False
    
    print("Searching for algorithm folders and CSV files...")
    
    # Search in each possible path
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        print(f"Checking in: {search_path}")
        
        # Look for algorithm folders
        for item in os.listdir(search_path):
            item_path = os.path.join(search_path, item)
            
            # Skip if not a directory
            if not os.path.isdir(item_path):
                continue
                
            # Check if folder name matches any algorithm
            algo_found = None
            for algo in algorithms_caps:
                if algo.lower() in item.lower():
                    algo_found = algo.lower()
                    break
            
            if not algo_found:
                continue
                
            print(f"Found algorithm folder: {item} -> {algo_found}")
            
            # Search for CSV files recursively
            csv_files = []
            for root, _, files in os.walk(item_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            print(f"  Found {len(csv_files)} CSV files in {algo_found} folder")
            
            # Process each CSV file
            for file in csv_files:
                filename = os.path.basename(file)
                
                # Try to determine place from filename
                place_found = None
                for place in ['JK2', 'MKS', 'SBY']:
                    if place in filename:
                        place_found = place
                        break
                
                if not place_found:
                    print(f"  Skipping file {filename} - can't determine place")
                    continue
                
                # Try to determine instance ID from filename
                # This is a simplified example - adjust for your actual file naming convention
                instance_id = filename.replace('.csv', '')
                
                # Load the CSV file
                try:
                    # Assuming the CSV has no header and two columns: cost,time
                    df = pd.read_csv(file, header=None, names=['cost', 'time'])
                    
                    # If multiple rows in a file, use the file as the instance_id
                    # Store each instance with its cost and time
                    for i, row in df.iterrows():
                        # Create a unique instance ID for each row
                        row_instance_id = f"{instance_id}_{i}" if df.shape[0] > 1 else instance_id
                        
                        # Add to instance_ids set for this place
                        data[place_found]['instance_ids'].add(row_instance_id)
                        
                        # Initialize dictionaries if needed
                        if algo_found not in data[place_found]['cost']:
                            data[place_found]['cost'][algo_found] = {}
                            data[place_found]['time'][algo_found] = {}
                        
                        # Store the data with instance_id as key
                        data[place_found]['cost'][algo_found][row_instance_id] = row['cost']
                        data[place_found]['time'][algo_found][row_instance_id] = row['time']
                    
                    print(f"  Loaded {df.shape[0]} rows from {filename} for {place_found}, {algo_found}")
                    found_files = True
                except Exception as e:
                    print(f"  Error loading {filename}: {str(e)}")
    
    if not found_files:
        print("\nNo CSV files were found! Expected folder structure:")
        print("root_folder/")
        print("    ga/")
        print("        JK2_*.csv")
        print("        MKS_*.csv")
        print("        SBY_*.csv")
        print("    de/")
        print("        JK2_*.csv")
        print("        ...")
        print("    ...")
        print("\nOR")
        print("root_folder/")
        print("    results/")
        print("        ga/")
        print("            JK2_*.csv")
        print("            ...")
    
    # Data availability report
    print("\nData Availability Report:")
    for place in data:
        print(f"\n{place}:")
        instances = data[place]['instance_ids']
        print(f"  Total unique problem instances: {len(instances)}")
        
        for metric in ['cost', 'time']:
            print(f"  {metric.capitalize()}:")
            for algo in algorithms:
                if algo in data[place][metric]:
                    instance_count = len(data[place][metric][algo])
                    print(f"    {algo}: {instance_count} instances")
                    
                    # Flag if insufficient data points
                    if instance_count < 2:
                        print(f"    WARNING: {algo} has fewer than 2 data points for {place}, {metric}")
                else:
                    print(f"    {algo}: Not found")
    
    return data

def prepare_paired_data(data):
    """
    Prepare paired data for analysis by finding common problem instances across algorithms.
    
    Args:
        data: Dictionary with data organized by place, metric, and algorithm
        
    Returns:
        Dictionary with paired data for each place, metric, and algorithm pair
    """
    paired_data = {}
    
    for place in data:
        paired_data[place] = {'cost': {}, 'time': {}}
        
        for metric in ['cost', 'time']:
            # Get all algorithms that have data for this place and metric
            available_algos = list(data[place][metric].keys())
            
            if len(available_algos) < 2:
                print(f"Not enough algorithms for {place}, {metric}. Found: {available_algos}")
                continue
            
            # Find all combinations of algorithms
            for algo1, algo2 in combinations(available_algos, 2):
                # Find instances common to both algorithms
                instances1 = set(data[place][metric][algo1].keys())
                instances2 = set(data[place][metric][algo2].keys())
                common_instances = instances1.intersection(instances2)
                
                if len(common_instances) < 2:
                    print(f"Insufficient common instances for {place}, {metric}, {algo1} vs {algo2}: {len(common_instances)}")
                    continue
                
                # Create paired data
                pairs = []
                for instance in sorted(common_instances):
                    pairs.append((
                        data[place][metric][algo1][instance],
                        data[place][metric][algo2][instance]
                    ))
                
                paired_data[place][metric][f"{algo1} vs {algo2}"] = {
                    'pairs': pairs,
                    'instances': sorted(common_instances)
                }
                
                print(f"Created {len(pairs)} paired observations for {place}, {metric}, {algo1} vs {algo2}")
    
    return paired_data

def perform_wilcoxon_signed_rank_test(paired_data):
    """
    Perform Wilcoxon signed-rank test on paired data.
    
    Args:
        paired_data: Dictionary with paired data for each place, metric, and algorithm pair
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    for place in paired_data:
        results[place] = {'cost': {}, 'time': {}}
        
        for metric in ['cost', 'time']:
            if not paired_data[place][metric]:
                print(f"No paired data available for {place}, {metric}")
                continue
                
            print(f"\nPerforming Wilcoxon signed-rank tests for {place}, {metric}:")
            
            for pair, data in paired_data[place][metric].items():
                algo1, algo2 = pair.split(" vs ")
                pairs = data['pairs']
                
                # Convert pairs to separate arrays
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                
                # Perform Wilcoxon signed-rank test (paired)
                try:
                    stat, p_value = wilcoxon(x, y, alternative='two-sided')
                    
                    # Determine which algorithm is better based on median difference
                    diff = x - y  # positive diff means x (algo1) has higher values
                    median_diff = np.median(diff)
                    better = algo2 if median_diff > 0 else algo1
                    
                    results[place][metric][pair] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'median_diff': median_diff,
                        'better': better,
                        'sample_size': len(pairs)
                    }
                    
                    print(f"  {pair}: p-value = {p_value:.4f}, {'significant' if p_value < 0.05 else 'not significant'}, better: {better}")
                    
                except Exception as e:
                    results[place][metric][pair] = {
                        'error': str(e)
                    }
                    print(f"  Error testing {pair}: {str(e)}")
    
    return results

def perform_paired_permutation_test(paired_data, n_permutations=10000):
    """
    Perform paired permutation test on paired data.
    
    Args:
        paired_data: Dictionary with paired data for each place, metric, and algorithm pair
        n_permutations: Number of permutations for the test
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    for place in paired_data:
        results[place] = {'cost': {}, 'time': {}}
        
        for metric in ['cost', 'time']:
            if not paired_data[place][metric]:
                print(f"No paired data available for {place}, {metric}")
                continue
                
            print(f"\nPerforming paired permutation tests for {place}, {metric}:")
            
            for pair, data in paired_data[place][metric].items():
                algo1, algo2 = pair.split(" vs ")
                pairs = data['pairs']
                
                # Convert pairs to separate arrays
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                
                # Define test statistic: difference in means
                def statistic(x, y):
                    return np.mean(x) - np.mean(y)
                
                # Perform paired permutation test
                try:
                    permutation_result = permutation_test(
                        (x, y), 
                        statistic, 
                        vectorized=False,
                        n_resamples=n_permutations, 
                        alternative='two-sided', 
                        permutation_type='samples'
                    )
                    
                    p_value = permutation_result.pvalue
                    
                    # Determine which algorithm is better based on mean difference
                    mean_diff = np.mean(x) - np.mean(y)
                    better = algo2 if mean_diff > 0 else algo1
                    
                    results[place][metric][pair] = {
                        'statistic': statistic(x, y),
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_diff': mean_diff,
                        'better': better,
                        'sample_size': len(pairs)
                    }
                    
                    print(f"  {pair}: p-value = {p_value:.4f}, {'significant' if p_value < 0.05 else 'not significant'}, better: {better}")
                    
                except Exception as e:
                    results[place][metric][pair] = {
                        'error': str(e)
                    }
                    print(f"  Error testing {pair}: {str(e)}")
    
    return results

def visualize_results(wilcoxon_results, permutation_results):
    """
    Create visualizations to compare Wilcoxon and permutation test results.
    
    Args:
        wilcoxon_results: Dictionary with Wilcoxon signed-rank test results
        permutation_results: Dictionary with permutation test results
    """
    for place in wilcoxon_results:
        for metric in ['cost', 'time']:
            if not wilcoxon_results[place][metric] or not permutation_results[place][metric]:
                continue
                
            # Prepare data for plotting
            pairs = []
            wilcoxon_pvals = []
            permutation_pvals = []
            significant = []
            
            for pair in wilcoxon_results[place][metric]:
                if pair in permutation_results[place][metric]:
                    if 'p_value' in wilcoxon_results[place][metric][pair] and 'p_value' in permutation_results[place][metric][pair]:
                        pairs.append(pair)
                        wilcoxon_pvals.append(wilcoxon_results[place][metric][pair]['p_value'])
                        permutation_pvals.append(permutation_results[place][metric][pair]['p_value'])
                        significant.append(wilcoxon_results[place][metric][pair]['significant'] or 
                                          permutation_results[place][metric][pair]['significant'])
            
            if not pairs:
                continue
                
            # Create bar plot comparing p-values
            plt.figure(figsize=(12, 6))
            x = np.arange(len(pairs))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, wilcoxon_pvals, width, label='Wilcoxon Signed-Rank Test')
            bars2 = plt.bar(x + width/2, permutation_pvals, width, label='Paired Permutation Test')
            
            # Add horizontal line at p=0.05
            plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Significance Level (p=0.05)')
            
            plt.xlabel('Algorithm Pairs')
            plt.ylabel('p-value')
            plt.title(f'Comparison of p-values for {place} - {metric.capitalize()}')
            plt.xticks(x, [p.replace(' vs ', '\nvs\n') for p in pairs], rotation=0)
            plt.legend()
            
            # Highlight significant results
            for i, is_sig in enumerate(significant):
                if is_sig:
                    plt.annotate('*', xy=(x[i], 0.01), xytext=(0, 0), 
                                textcoords="offset points", ha='center', va='bottom',
                                fontsize=20, color='green')
            
            plt.tight_layout()
            plt.savefig(f'pvalue_comparison_{place}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plot of p-values for comparison
            plt.figure(figsize=(8, 8))
            plt.scatter(wilcoxon_pvals, permutation_pvals, alpha=0.7)
            
            # Add diagonal line
            lims = [0, max(max(wilcoxon_pvals), max(permutation_pvals)) * 1.1]
            plt.plot(lims, lims, 'r--')
            
            # Add vertical and horizontal lines at p=0.05
            plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
            
            # Label points with algorithm pairs
            for i, pair in enumerate(pairs):
                plt.annotate(pair, (wilcoxon_pvals[i], permutation_pvals[i]), 
                            fontsize=8, alpha=0.8)
            
            plt.xlabel('Wilcoxon Signed-Rank Test p-value')
            plt.ylabel('Paired Permutation Test p-value')
            plt.title(f'Comparison of Test Methods for {place} - {metric.capitalize()}')
            
            plt.tight_layout()
            plt.savefig(f'pvalue_scatter_{place}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

def visualize_paired_differences(paired_data):
    """
    Create visualizations of the paired differences.
    
    Args:
        paired_data: Dictionary with paired data for each place, metric, and algorithm pair
    """
    for place in paired_data:
        for metric in ['cost', 'time']:
            if not paired_data[place][metric]:
                continue
                
            n_pairs = len(paired_data[place][metric])
            if n_pairs == 0:
                continue
                
            # Create a grid of plots
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            fig.suptitle(f'Paired Differences for {place} - {metric.capitalize()}')
            
            # Flatten axes array if it's multi-dimensional
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Create plots for each pair
            for i, (pair, data) in enumerate(paired_data[place][metric].items()):
                if i >= len(axes):
                    break
                    
                algo1, algo2 = pair.split(" vs ")
                pairs = data['pairs']
                
                # Convert pairs to separate arrays
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                diff = x - y  # positive means algo1 has higher values
                
                # Create histogram of differences
                axes[i].hist(diff, bins=20, alpha=0.7)
                axes[i].axvline(x=0, color='r', linestyle='--')
                axes[i].set_title(f'{algo1} vs {algo2}')
                axes[i].set_xlabel(f'Difference ({algo1} - {algo2})')
                axes[i].set_ylabel('Frequency')
                
                # Add median line
                median_diff = np.median(diff)
                axes[i].axvline(x=median_diff, color='g', linestyle='-', label=f'Median: {median_diff:.2f}')
                axes[i].legend()
            
            # Hide any unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            plt.savefig(f'paired_differences_{place}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plots to compare algorithms directly
            for pair, data in paired_data[place][metric].items():
                algo1, algo2 = pair.split(" vs ")
                pairs = data['pairs']
                
                # Convert pairs to separate arrays
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                
                plt.figure(figsize=(8, 8))
                plt.scatter(x, y, alpha=0.7)
                
                # Add diagonal line (x=y)
                lims = [
                    min(min(x), min(y)) * 0.9,
                    max(max(x), max(y)) * 1.1
                ]
                plt.plot(lims, lims, 'r--', label='x=y')
                
                plt.xlabel(f'{algo1}')
                plt.ylabel(f'{algo2}')
                plt.title(f'Comparison of {algo1} vs {algo2} for {place} - {metric.capitalize()}')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'scatter_{place}_{metric}_{algo1}_vs_{algo2}.png', dpi=300, bbox_inches='tight')
                plt.close()

def create_summary_table(wilcoxon_results, permutation_results):
    """
    Create a summary table of the test results.
    
    Args:
        wilcoxon_results: Dictionary with Wilcoxon signed-rank test results
        permutation_results: Dictionary with permutation test results
        
    Returns:
        DataFrame with summary information
    """
    # Create lists to store rows for the table
    rows = []
    
    for place in wilcoxon_results:
        for metric in ['cost', 'time']:
            if not wilcoxon_results[place][metric]:
                continue
                
            for pair in wilcoxon_results[place][metric]:
                if 'p_value' not in wilcoxon_results[place][metric][pair]:
                    continue
                    
                # Extract Wilcoxon results
                w_p_value = wilcoxon_results[place][metric][pair]['p_value']
                w_significant = wilcoxon_results[place][metric][pair]['significant']
                w_better = wilcoxon_results[place][metric][pair].get('better', 'N/A')
                
                # Extract permutation results if available
                p_p_value = 'N/A'
                p_significant = False
                p_better = 'N/A'
                
                if (pair in permutation_results[place][metric] and 
                    'p_value' in permutation_results[place][metric][pair]):
                    p_p_value = permutation_results[place][metric][pair]['p_value']
                    p_significant = permutation_results[place][metric][pair]['significant']
                    p_better = permutation_results[place][metric][pair].get('better', 'N/A')
                
                # Check if results agree
                agree_significance = ((w_significant and p_significant) or 
                                      (not w_significant and not p_significant))
                agree_better = w_better == p_better
                
                # Add row to table
                rows.append({
                    'Place': place,
                    'Metric': metric.capitalize(),
                    'Comparison': pair,
                    'Wilcoxon p-value': w_p_value,
                    'Wilcoxon Significant': w_significant,
                    'Wilcoxon Better': w_better,
                    'Permutation p-value': p_p_value if p_p_value != 'N/A' else np.nan,
                    'Permutation Significant': p_significant,
                    'Permutation Better': p_better,
                    'Agree on Significance': agree_significance,
                    'Agree on Better Algorithm': agree_better
                })
    
    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows)
        
        # Sort by place, metric, and p-value
        df = df.sort_values(['Place', 'Metric', 'Wilcoxon p-value'])
        
        return df
    else:
        return pd.DataFrame()

def summarize_results(wilcoxon_results, permutation_results, summary_df):
    """
    Create a summary of the test results.
    
    Args:
        wilcoxon_results: Dictionary with Wilcoxon signed-rank test results
        permutation_results: Dictionary with permutation test results
        summary_df: DataFrame with summary information
        
    Returns:
        Summary of results as a string
    """
    summary = "Summary of Paired Statistical Tests\n"
    summary += "================================\n\n"
    
    # Add explanatory text
    summary += "This analysis compares the performance of different algorithms "
    summary += "using two types of paired statistical tests:\n"
    summary += "1. Wilcoxon Signed-Rank Test: A non-parametric test for paired samples\n"
    summary += "2. Paired Permutation Test: A randomization-based test for paired samples\n\n"
    
    # Add overview of available data
    available_data = {}
    for place in wilcoxon_results:
        available_data[place] = {'cost': set(), 'time': set()}
        for metric in ['cost', 'time']:
            for pair in wilcoxon_results[place][metric]:
                if 'sample_size' in wilcoxon_results[place][metric][pair]:
                    algo1, algo2 = pair.split(" vs ")
                    available_data[place][metric].add(algo1)
                    available_data[place][metric].add(algo2)
    
    summary += "Available Data Overview:\n"
    summary += "----------------------\n"
    for place in available_data:
        summary += f"{place}:\n"
        for metric in ['cost', 'time']:
            algos = available_data[place][metric]
            if algos:
                summary += f"  {metric.capitalize()}: {', '.join(sorted(algos))}\n"
            else:
                summary += f"  {metric.capitalize()}: No data available\n"
        summary += "\n"
    
    # Add test results by place and metric
    for place in wilcoxon_results:
        summary += f"Results for {place}:\n"
        summary += "=" * (11 + len(place)) + "\n\n"
        
        for metric in ['cost', 'time']:
            if not wilcoxon_results[place][metric]:
                summary += f"{metric.capitalize()}: No test results available\n\n"
                continue
                
            summary += f"{metric.capitalize()} Analysis:\n"
            summary += "-" * (len(metric) + 10) + "\n\n"
            
            # Filter summary_df for this place and metric
            place_metric_df = summary_df[(summary_df['Place'] == place) & 
                                        (summary_df['Metric'] == metric.capitalize())]
            
            if place_metric_df.empty:
                summary += "No valid test results available\n\n"
                continue
            
            # Count significant results
            sig_count = place_metric_df['Wilcoxon Significant'].sum()
            total_count = len(place_metric_df)
            
            summary += f"Found {sig_count} significant differences out of {total_count} comparisons.\n\n"
            
            # Add table of results
            if not place_metric_df.empty:
                # Format p-values in the DataFrame
                place_metric_df_formatted = place_metric_df.copy()
                place_metric_df_formatted['Wilcoxon p-value'] = place_metric_df_formatted['Wilcoxon p-value'].map('{:.4f}'.format)
                
                # Format permutation p-values, handling 'N/A' values
                def format_p_value(val):
                    if pd.isna(val):
                        return 'N/A'
                    else:
                        return f'{val:.4f}'
                        
                place_metric_df_formatted['Permutation p-value'] = place_metric_df_formatted['Permutation p-value'].apply(format_p_value)
                
                # Select and reorder columns for the summary
                columns_to_show = ['Comparison', 'Wilcoxon p-value', 'Wilcoxon Significant', 
                                  'Wilcoxon Better', 'Permutation p-value', 'Permutation Significant']
                table_str = place_metric_df_formatted[columns_to_show].to_string(index=False)
                summary += table_str + "\n\n"
            
            # Add information about algorithms performance
            summary += "Algorithm Performance Ranking:\n"
            
            # Count how many times each algorithm is "better"
            better_counts = {}
            for _, row in place_metric_df.iterrows():
                better_algo = row['Wilcoxon Better']
                if better_algo not in better_counts:
                    better_counts[better_algo] = 0
                better_counts[better_algo] += 1
            
            # Sort algorithms by better count
            ranked_algos = sorted(better_counts.items(), key=lambda x: x[1], reverse=True)
            
            for i, (algo, count) in enumerate(ranked_algos):
                summary += f"{i+1}. {algo}: Better in {count} comparisons\n"
            
            summary += "\n\n"
    
    # Add overall conclusions
    summary += "Overall Conclusions:\n"
    summary += "------------------\n"
    
    # Check if there's agreement between test methods
    agree_sig = summary_df['Agree on Significance'].mean() if not summary_df.empty else 0
    agree_better = summary_df['Agree on Better Algorithm'].mean() if not summary_df.empty else 0
    
    summary += f"The Wilcoxon Signed-Rank Test and Paired Permutation Test agreed on:\n"
    summary += f"- Statistical significance in {agree_sig:.1%} of comparisons\n"
    summary += f"- Which algorithm performed better in {agree_better:.1%} of comparisons\n\n"
    
    # Identify best algorithm overall by location
    summary += "Best performing algorithms by location:\n"
    for place in wilcoxon_results:
        best_algos = {'cost': 'Unknown', 'time': 'Unknown'}
        
        for metric in ['cost', 'time']:
            # Filter for this place and metric
            place_metric_df = summary_df[(summary_df['Place'] == place) & 
                                        (summary_df['Metric'] == metric.capitalize())]
            
            if not place_metric_df.empty:
                # Count "better" occurrences
                better_counts = {}
                for _, row in place_metric_df.iterrows():
                    if row['Wilcoxon Significant']:
                        better_algo = row['Wilcoxon Better']
                        if better_algo not in better_counts:
                            better_counts[better_algo] = 0
                        better_counts[better_algo] += 1
                
                # Find algorithm with most "better" occurrences
                if better_counts:
                    best_algos[metric] = max(better_counts.items(), key=lambda x: x[1])[0]
        
        summary += f"- {place}: Best for cost: {best_algos['cost']}, Best for time: {best_algos['time']}\n"
    
    return summary

def create_latex_tables(summary_df):
    """
    Create LaTeX tables for the results.
    
    Args:
        summary_df: DataFrame with summary information
        
    Returns:
        String with LaTeX tables
    """
    latex_output = "% LaTeX Tables for Statistical Test Results\n\n"
    
    for place in summary_df['Place'].unique():
        for metric in ['Cost', 'Time']:
            # Filter for this place and metric
            place_metric_df = summary_df[(summary_df['Place'] == place) & 
                                        (summary_df['Metric'] == metric)]
            
            if place_metric_df.empty:
                continue
            
            # Format p-values
            place_metric_df_formatted = place_metric_df.copy()
            place_metric_df_formatted['Wilcoxon p-value'] = place_metric_df_formatted['Wilcoxon p-value'].map('{:.4f}'.format)
            
            # Format permutation p-values, handling 'N/A' values
            def format_p_value(val):
                if pd.isna(val):
                    return 'N/A'
                else:
                    return f'{val:.4f}'
                    
            place_metric_df_formatted['Permutation p-value'] = place_metric_df_formatted['Permutation p-value'].apply(format_p_value)
            
            # Add significance markers
            def add_sig_marker(row, test_type):
                p_val_col = f'{test_type} p-value'
                sig_col = f'{test_type} Significant'
                
                p_val = row[p_val_col]
                if p_val == 'N/A':
                    return p_val
                
                if row[sig_col]:
                    return f"{p_val}$^*$"
                else:
                    return p_val
            
            place_metric_df_formatted['Wilcoxon p-value'] = place_metric_df_formatted.apply(
                lambda row: add_sig_marker(row, 'Wilcoxon'), axis=1
            )
            
            place_metric_df_formatted['Permutation p-value'] = place_metric_df_formatted.apply(
                lambda row: add_sig_marker(row, 'Permutation'), axis=1
            )
            
            # Create LaTeX table
            latex_output += f"% Table for {place} - {metric}\n"
            latex_output += "\\begin{table}[htbp]\n"
            latex_output += "\\centering\n"
            latex_output += f"\\caption{{Statistical test results for {place} - {metric}}}\n"
            latex_output += "\\begin{tabular}{lcccc}\n"
            latex_output += "\\hline\n"
            latex_output += "Comparison & Wilcoxon p-value & Wilcoxon Better & Permutation p-value & Permutation Better \\\\ \n"
            latex_output += "\\hline\n"
            
            # Add rows
            for _, row in place_metric_df_formatted.iterrows():
                latex_output += f"{row['Comparison']} & {row['Wilcoxon p-value']} & {row['Wilcoxon Better']} & "
                latex_output += f"{row['Permutation p-value']} & {row['Permutation Better']} \\\\ \n"
            
            latex_output += "\\hline\n"
            latex_output += "\\end{tabular}\n"
            latex_output += "\\label{tab:stats_" + f"{place.lower()}_{metric.lower()}" + "}\n"
            latex_output += "\\end{table}\n\n"
    
    return latex_output

def main():
    # Get folder path from user or use current directory
    folder_path = input("Enter the path to the folder containing algorithm folders (press Enter to use current directory): ")
    if not folder_path:
        folder_path = '.'
    
    # Load data
    print("\n1. Loading data...")
    data = load_data(folder_path)
    
    # Check if we have data
    empty_data = True
    for place in data:
        for metric in ['cost', 'time']:
            if data[place][metric]:
                empty_data = False
                break
        if not empty_data:
            break
    
    if empty_data:
        print("\nNo data found. Please check the folder structure and try again.")
        return
    
    # Prepare paired data
    print("\n2. Preparing paired data...")
    paired_data = prepare_paired_data(data)
    
    # Perform Wilcoxon signed-rank tests
    print("\n3. Performing Wilcoxon signed-rank tests...")
    wilcoxon_results = perform_wilcoxon_signed_rank_test(paired_data)
    
    # Perform paired permutation tests
    print("\n4. Performing paired permutation tests...")
    try:
        permutation_results = perform_paired_permutation_test(paired_data)
    except ImportError:
        print("Warning: SciPy's permutation_test function not available (requires SciPy >= 1.7.0).")
        print("Skipping permutation tests.")
        permutation_results = {place: {'cost': {}, 'time': {}} for place in paired_data}
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    
    # Visualize paired differences
    try:
        visualize_paired_differences(paired_data)
    except Exception as e:
        print(f"Warning: Error creating paired difference visualizations: {str(e)}")
    
    # Compare test results
    try:
        visualize_results(wilcoxon_results, permutation_results)
    except Exception as e:
        print(f"Warning: Error creating test comparison visualizations: {str(e)}")
    
    # Create summary table
    print("\n6. Creating summary table...")
    summary_df = create_summary_table(wilcoxon_results, permutation_results)
    
    # Generate LaTeX tables
    latex_tables = create_latex_tables(summary_df)
    with open('statistical_tests_tables.tex', 'w') as f:
        f.write(latex_tables)
    
    # Generate summary
    print("\n7. Generating summary report...")
    summary = summarize_results(wilcoxon_results, permutation_results, summary_df)
    
    # Save summary to file
    with open('paired_tests_summary.txt', 'w') as f:
        f.write(summary)
    
    # Save summary table to CSV
    if not summary_df.empty:
        summary_df.to_csv('statistical_tests_summary.csv', index=False)
    
    print("\nAnalysis complete!")
    print("Summary saved to 'paired_tests_summary.txt'")
    print("Summary table saved to 'statistical_tests_summary.csv'")
    print("LaTeX tables saved to 'statistical_tests_tables.tex'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()