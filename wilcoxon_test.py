import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def load_data(folder_path='.'):
    """
    Load CSV files from algorithm-specific folders.
    - Algorithms (GA, DE, BRKGA, PSO, AVNS) are determined by folder names
    - Locations (JK2, MKS, SBY) are found within the filenames
    Each file contains rows with format: total_cost,running_time
    
    Args:
        folder_path: Path to the root folder containing algorithm subfolders
        
    Returns:
        Dictionary with data organized by place, algorithm, and metric (cost/time)
    """
    data = {
        'JK2': {'cost': {}, 'time': {}},
        'MKS': {'cost': {}, 'time': {}},
        'SBY': {'cost': {}, 'time': {}}
    }
    
    # List of places to look for in filenames
    places = ['JK2', 'MKS', 'SBY']
    
    # List of algorithm folders to search for
    algorithm_folders = {
        'ga': 'GA', 
        'de': 'DE', 
        'brkga': 'BRKGA', 
        'pso': 'PSO', 
        'avns': 'AVNS'
    }
    
    # Also look for capitalized versions of the folder names
    for key in list(algorithm_folders.keys()):
        algorithm_folders[key.upper()] = algorithm_folders[key]
    
    # Look for algorithm folders in the main path or in a 'results' subfolder
    possible_paths = [
        folder_path,  # Direct path
        os.path.join(folder_path, 'results')  # With 'results' subfolder
    ]
    
    files_processed = 0
    algorithms_found = set()
    places_found = set()
    
    # Process each algorithm folder
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
            
        print(f"Searching for algorithm folders in: {base_path}")
        
        # List all subdirectories in the base path
        try:
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            print(f"Found subdirectories: {subdirs}")
        except Exception as e:
            print(f"Error listing subdirectories: {str(e)}")
            subdirs = []
        
        for algo_folder_name in subdirs:
            # Check if this folder name corresponds to an algorithm
            algo_name = None
            for folder_pattern, algo_id in algorithm_folders.items():
                if folder_pattern in algo_folder_name.lower():
                    algo_name = algo_id
                    break
            
            if not algo_name:
                print(f"Skipping folder {algo_folder_name} - not recognized as an algorithm folder")
                continue
                
            algo_path = os.path.join(base_path, algo_folder_name)
            print(f"Processing algorithm folder: {algo_path} (Algorithm: {algo_name})")
            algorithms_found.add(algo_name)
            
            # Get all CSV files in this algorithm folder (including subdirectories)
            csv_files = glob.glob(os.path.join(algo_path, '**', '*.csv'), recursive=True)
            print(f"Found {len(csv_files)} CSV files in {algo_name} folder")
            
            # Process each CSV file
            for file in csv_files:
                filename = os.path.basename(file)
                
                # Try to determine place from filename
                place_found = None
                for place in places:
                    if place in filename:
                        place_found = place
                        break
                
                if not place_found:
                    print(f"Skipping file {filename} - can't determine place")
                    continue
                
                places_found.add(place_found)
                
                # Load the CSV file
                try:
                    # Assuming the CSV has no header and two columns: cost,time
                    df = pd.read_csv(file, header=None, names=['cost', 'time'])
                    
                    # Store the data
                    data[place_found]['cost'][algo_name] = df['cost'].values
                    data[place_found]['time'][algo_name] = df['time'].values
                    
                    print(f"Loaded {len(df)} rows from {filename} for {place_found}, {algo_name}")
                    files_processed += 1
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
    
    print(f"Total files processed: {files_processed}")
    if files_processed == 0:
        print("WARNING: No files were processed. Please check the folder structure.")
    
    # Print a summary of what was found
    print("\nData loading summary:")
    print(f"Algorithms found: {', '.join(sorted(algorithms_found))}")
    print(f"Places found: {', '.join(sorted(places_found))}")
    
    for place in data:
        for metric in ['cost', 'time']:
            algos = list(data[place][metric].keys())
            if algos:
                print(f"{place} - {metric}: Found data for {len(algos)} algorithms: {', '.join(algos)}")
                for algo in algos:
                    count = len(data[place][metric][algo])
                    print(f"  {algo}: {count} data points{'  [INSUFFICIENT]' if count < 2 else ''}")
            else:
                print(f"{place} - {metric}: No data found")
    
    return data

def perform_wilcoxon_tests(data):
    """
    Perform Wilcoxon Rank Sum Test (Mann-Whitney U) between all pairs of algorithms 
    for each place and metric (cost/time).
    
    Args:
        data: Dictionary with data organized by place, metric, and algorithm
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    for place in data:
        results[place] = {'cost': {}, 'time': {}}
        
        for metric in ['cost', 'time']:
            algorithms = list(data[place][metric].keys())
            
            if len(algorithms) < 2:
                print(f"WARN: Cannot perform Wilcoxon test for {place} - {metric}: Need at least 2 algorithms (found {len(algorithms)})")
                continue
                
            print(f"Performing tests for {place} - {metric} with {len(algorithms)} algorithms: {', '.join(algorithms)}")
            
            # Count how many tests we'll be performing
            test_count = 0
            skipped_count = 0
            
            for algo1, algo2 in combinations(algorithms, 2):
                # Check if there's enough data
                if (len(data[place][metric][algo1]) > 1 and 
                    len(data[place][metric][algo2]) > 1):
                    test_count += 1
                else:
                    skipped_count += 1
            
            if test_count == 0:
                print(f"WARN: No tests will be performed for {place} - {metric} (insufficient data points)")
                continue
                
            print(f"Will perform {test_count} tests ({skipped_count} pairs skipped due to insufficient data)")
            
            for algo1, algo2 in combinations(algorithms, 2):
                # Check if there's enough data
                if (len(data[place][metric][algo1]) > 1 and 
                    len(data[place][metric][algo2]) > 1):
                    # Perform Wilcoxon Rank Sum Test (Mann-Whitney U)
                    try:
                        stat, p_value = mannwhitneyu(
                            data[place][metric][algo1], 
                            data[place][metric][algo2],
                            alternative='two-sided'
                        )
                        results[place][metric][f"{algo1} vs {algo2}"] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'better': algo1 if np.median(data[place][metric][algo1]) < np.median(data[place][metric][algo2]) else algo2
                        }
                        print(f"Test {algo1} vs {algo2}: p-value = {p_value:.5f} {'(significant)' if p_value < 0.05 else ''}")
                    except Exception as e:
                        results[place][metric][f"{algo1} vs {algo2}"] = {
                            'error': str(e)
                        }
                        print(f"Error testing {algo1} vs {algo2}: {str(e)}")
    
    # Print a summary of the test results
    print("\nWilcoxon test summary:")
    for place in results:
        for metric in ['cost', 'time']:
            if metric in results[place] and results[place][metric]:
                test_count = len(results[place][metric])
                significant_count = sum(1 for result in results[place][metric].values() 
                                      if 'significant' in result and result['significant'])
                print(f"{place} - {metric}: {test_count} tests, {significant_count} significant differences")
            else:
                print(f"{place} - {metric}: No tests performed")
    
    return results

def create_p_value_matrix(results):
    """
    Create p-value matrices for each place and metric.
    
    Args:
        results: Dictionary with test results
        
    Returns:
        Dictionary with p-value matrices
    """
    matrices = {}
    
    for place in results:
        matrices[place] = {}
        
        for metric in ['cost', 'time']:
            if not results[place][metric]:  # Skip if no results for this metric
                continue
                
            algorithms = set()
            for pair in results[place][metric]:
                algo1, algo2 = pair.split(" vs ")
                algorithms.add(algo1)
                algorithms.add(algo2)
            
            algorithms = sorted(list(algorithms))
            n = len(algorithms)
            
            if n == 0:
                continue
            
            # Create a matrix filled with NaN
            matrix = np.full((n, n), np.nan)
            
            # Fill the matrix with p-values
            for i, algo1 in enumerate(algorithms):
                for j, algo2 in enumerate(algorithms):
                    if i == j:
                        matrix[i, j] = 1.0  # Same algorithm, p-value = 1
                    else:
                        key = f"{algo1} vs {algo2}"
                        alt_key = f"{algo2} vs {algo1}"
                        
                        if key in results[place][metric] and 'p_value' in results[place][metric][key]:
                            matrix[i, j] = results[place][metric][key]['p_value']
                        elif alt_key in results[place][metric] and 'p_value' in results[place][metric][alt_key]:
                            matrix[i, j] = results[place][metric][alt_key]['p_value']
            
            matrices[place][metric] = pd.DataFrame(matrix, index=algorithms, columns=algorithms)
    
    return matrices

def visualize_results(matrices):
    """
    Visualize p-value matrices as heatmaps.
    
    Args:
        matrices: Dictionary with p-value matrices
    """
    for place in matrices:
        for metric in matrices[place]:
            matrix = matrices[place][metric]
            
            plt.figure(figsize=(10, 8))
            
            sns.heatmap(matrix, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.1, 
                       linewidths=0.5, cbar=True, fmt='.4f')
            
            plt.title(f'P-values for {place} - {metric.capitalize()}')
            plt.ylabel('Algorithm')
            plt.xlabel('Algorithm')
            
            # Highlight significant p-values (p < 0.05)
            for x in range(matrix.shape[0]):
                for y in range(matrix.shape[1]):
                    if pd.notna(matrix.iloc[x, y]) and matrix.iloc[x, y] < 0.05 and x != y:
                        plt.gca().add_patch(plt.Rectangle((y, x), 1, 1, fill=False, edgecolor='black', lw=2))
            
            plt.tight_layout()
            plt.savefig(f'wilcoxon_{place}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

def create_boxplots(data):
    """
    Create boxplots to visualize the distribution of cost and time for each algorithm at each place.
    
    Args:
        data: Dictionary with data organized by place, metric, and algorithm
    """
    for place in data:
        for metric in ['cost', 'time']:
            # Prepare data for boxplot
            df_list = []
            
            for algo in data[place][metric]:
                if len(data[place][metric][algo]) > 0:
                    temp_df = pd.DataFrame({
                        'Algorithm': algo,
                        'Value': data[place][metric][algo]
                    })
                    df_list.append(temp_df)
            
            if not df_list:
                continue
                
            df = pd.concat(df_list, ignore_index=True)
            
            # Create boxplot
            plt.figure(figsize=(12, 6))
            
            sns.boxplot(x='Algorithm', y='Value', data=df)
            
            plt.title(f'{metric.capitalize()} Distribution for {place}')
            plt.ylabel(f'{metric.capitalize()}')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f'boxplot_{place}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

def calculate_descriptive_stats(data):
    """
    Calculate descriptive statistics for each algorithm, place, and metric.
    
    Args:
        data: Dictionary with data organized by place, metric, and algorithm
        
    Returns:
        Dictionary with descriptive statistics
    """
    stats = {}
    
    for place in data:
        stats[place] = {'cost': {}, 'time': {}}
        
        for metric in ['cost', 'time']:
            for algo in data[place][metric]:
                values = data[place][metric][algo]
                
                if len(values) > 0:
                    stats[place][metric][algo] = {
                        'count': len(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan
                    }
    
    return stats

def summarize_results(results, stats):
    """
    Create a summary of the Wilcoxon test results and descriptive statistics.
    
    Args:
        results: Dictionary with test results
        stats: Dictionary with descriptive statistics
        
    Returns:
        Summary of results as a string
    """
    summary = "Summary of Wilcoxon Rank Sum Test Results:\n\n"
    
    for place in results:
        summary += f"Results for {place}:\n"
        summary += "=" * 50 + "\n\n"
        
        for metric in ['cost', 'time']:
            summary += f"{metric.upper()} ANALYSIS:\n"
            summary += "-" * 50 + "\n\n"
            
            # First add descriptive statistics
            summary += "Descriptive Statistics:\n"
            summary += "-" * 25 + "\n"
            
            if place in stats and metric in stats[place]:
                algo_stats = stats[place][metric]
                if algo_stats:
                    # Create a table header
                    summary += f"{'Algorithm':<10} {'Count':>8} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>12} {'Std':>12} {'CV':>8}\n"
                    
                    for algo, stat in algo_stats.items():
                        summary += f"{algo:<10} {stat['count']:>8} {stat['min']:>12.2f} {stat['max']:>12.2f} {stat['mean']:>12.2f} {stat['median']:>12.2f} {stat['std']:>12.2f} {stat['cv']:>8.2f}\n"
                else:
                    summary += "No data available\n"
            else:
                summary += "No data available\n"
            
            summary += "\n"
            
            # Then add Wilcoxon test results
            summary += "Statistical Test Results:\n"
            summary += "-" * 25 + "\n"
            
            if metric in results[place] and results[place][metric]:
                significant_tests = []
                non_significant_tests = []
                
                for pair, result in results[place][metric].items():
                    if 'p_value' in result:
                        p_value = result['p_value']
                        significant = result['significant']
                        better = result.get('better', 'N/A')
                        
                        result_str = f"{pair}: p-value = {p_value:.4f}"
                        
                        if significant:
                            result_str += f" (significant difference, {better} is better)"
                            significant_tests.append(result_str)
                        else:
                            result_str += " (no significant difference)"
                            non_significant_tests.append(result_str)
                    else:
                        non_significant_tests.append(f"{pair}: {result.get('error', 'Error occurred')}")
                
                summary += "Significant differences:\n"
                if significant_tests:
                    for test in significant_tests:
                        summary += f"  - {test}\n"
                else:
                    summary += "  None\n"
                
                summary += "\nNon-significant differences:\n"
                if non_significant_tests:
                    for test in non_significant_tests:
                        summary += f"  - {test}\n"
                else:
                    summary += "  None\n"
            else:
                summary += "No test results available\n"
            
            summary += "\n\n"
    
    return summary

def main():
    # Get folder path from user or use current directory
    folder_path = input("Enter the path to the root folder containing algorithm folders (press Enter to use current directory): ")
    if not folder_path:
        folder_path = '.'
    
    # Load data
    print("Loading data...")
    data = load_data(folder_path)
    
    # Check if we have data
    empty_data = True
    for place in data:
        for metric in data[place]:
            if data[place][metric]:
                empty_data = False
                break
        if not empty_data:
            break
    
    if empty_data:
        print("No data found. Please check the folder structure and try again.")
        print("Expected structure: root_folder/[results/][ga|de|brkga|pso|avns]/[any_path/]JK2_*.csv")
        return
    
    # Calculate descriptive statistics
    print("Calculating descriptive statistics...")
    stats = calculate_descriptive_stats(data)
    
    # Create boxplots
    print("Creating boxplots...")
    create_boxplots(data)
    
    # Perform Wilcoxon Rank Sum tests
    print("Performing Wilcoxon Rank Sum tests...")
    results = perform_wilcoxon_tests(data)
    
    # Create p-value matrices
    print("Creating p-value matrices...")
    matrices = create_p_value_matrix(results)
    
    # Visualize the results
    print("Visualizing test results...")
    visualize_results(matrices)
    
    # Generate summary
    print("Generating summary report...")
    summary = summarize_results(results, stats)
    
    # Save summary to file
    with open('wilcoxon_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\nAnalysis complete!")
    print("Summary saved to 'wilcoxon_summary.txt'")
    print("Heatmaps saved as 'wilcoxon_<place>_<metric>.png'")
    print("Boxplots saved as 'boxplot_<place>_<metric>.png'")
    
    # Print a brief overview
    print("\nBrief overview of findings:")
    for place in results:
        for metric in ['cost', 'time']:
            if metric in results[place] and results[place][metric]:
                significant_pairs = [pair for pair, result in results[place][metric].items() 
                                    if 'significant' in result and result['significant']]
                if significant_pairs:
                    print(f"{place} - {metric}: Found {len(significant_pairs)} significant differences")
                    # Print the best algorithm for this place and metric
                    try:
                        algo_scores = {}
                        for algo in data[place][metric].keys():
                            # Lower is better for both cost and time
                            algo_scores[algo] = np.median(data[place][metric][algo])
                        
                        best_algo = min(algo_scores, key=algo_scores.get)
                        print(f"  Best algorithm: {best_algo} (median {metric}: {algo_scores[best_algo]:.2f})")
                    except Exception as e:
                        print(f"  Could not determine best algorithm: {str(e)}")
                else:
                    print(f"{place} - {metric}: No significant differences found")

if __name__ == "__main__":
    main()