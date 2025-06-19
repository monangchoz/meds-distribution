import pandas as pd
import numpy as np
import os
import glob
import re
from scipy import stats
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def round_customer_count(num_customers):
    """Round customer count to research values: 15, 30, or 50"""
    if pd.isna(num_customers):
        return num_customers
    
    research_values = [15, 30, 50]
    closest = min(research_values, key=lambda x: abs(x - num_customers))
    return closest

def extract_detailed_info(filename):
    """Extract detailed information from filename"""
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
    
    # Extract ratios
    ratio_match = re.search(r'r_([\d.]+)_([\d.]+)_([\d.]+)', filename, re.IGNORECASE)
    ratios = None
    if ratio_match:
        ratios = (float(ratio_match.group(1)), 
                 float(ratio_match.group(2)), 
                 float(ratio_match.group(3)))
    
    return location, num_customers, num_clusters, num_vehicles, data_type, ratios

def load_data_for_testing(base_path):
    """Load CSV files for statistical testing"""
    enhanced_data = []
    
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
            
        csv_files = glob.glob(os.path.join(algorithm_path, "*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            location, num_customers, num_clusters, num_vehicles, data_type, ratios = extract_detailed_info(filename)
            
            try:
                df = pd.read_csv(file_path, header=None, names=['total_cost', 'running_time'])
                
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
                
                print(f"✓ Processed {filename} -> {algorithm_name}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    df = pd.DataFrame(enhanced_data)
    
    if not df.empty:
        # Create cargo mix category
        def categorize_cargo_mix(row):
            if pd.isna(row['ratio_small']):
                return 'Unknown'
            
            small, medium, large = row['ratio_small'], row['ratio_medium'], row['ratio_large']
            small_r, medium_r, large_r = round(small, 2), round(medium, 2), round(large, 2)
            
            if (small_r, medium_r, large_r) == (0.2, 0.2, 0.6):
                return '(0.2-0.2-0.6)'
            elif (small_r, medium_r, large_r) == (0.2, 0.6, 0.2):
                return '(0.2-0.6-0.2)'
            elif (small_r, medium_r, large_r) == (0.6, 0.2, 0.2):
                return '(0.6-0.2-0.2)'
            elif abs(small_r - 0.33) < 0.02 and abs(medium_r - 0.33) < 0.02 and abs(large_r - 0.33) < 0.02:
                return '(1/3-1/3-1/3)'
            else:
                return f'Other ({small_r}-{medium_r}-{large_r})'
        
        df['cargo_mix'] = df.apply(categorize_cargo_mix, axis=1)
        
        print(f"\n📊 Loaded {len(df)} records from {df['algorithm'].nunique()} algorithms")
        print(f"📍 Locations: {sorted(df['location'].dropna().unique())}")
        print(f"👥 Customer counts (rounded): {sorted(df['num_customers_rounded'].dropna().unique())}")
        print(f"🚛 Vehicle counts: {sorted(df['num_vehicles'].dropna().unique())}")
        print(f"📦 Cargo mixes: {sorted(df['cargo_mix'].unique())}")
    
    return df

def create_paired_dataset(df, baseline_alg='GA'):
    """Create paired dataset for statistical testing with proper 1:1 pairing"""
    if df.empty:
        return None
    
    # Create instance identifier
    df['instance_id'] = (
        df['location'].astype(str) + '_' +
        df['num_customers_rounded'].astype(str) + '_' +
        df['num_clusters'].astype(str) + '_' +
        df['num_vehicles'].astype(str) + '_' +
        df['data_type'].astype(str) + '_' +
        df['cargo_mix'].astype(str)
    )
    
    # Get baseline algorithm data
    baseline_data = df[df['algorithm'] == baseline_alg].copy()
    if baseline_data.empty:
        print(f"❌ No data found for baseline algorithm: {baseline_alg}")
        return None
    
    # Create paired comparisons
    paired_results = {}
    algorithms = [alg for alg in df['algorithm'].unique() if alg != baseline_alg]
    
    for compare_alg in algorithms:
        compare_data = df[df['algorithm'] == compare_alg].copy()
        
        # Find common instances
        baseline_instances = set(baseline_data['instance_id'])
        compare_instances = set(compare_data['instance_id'])
        common_instances = baseline_instances.intersection(compare_instances)
        
        if len(common_instances) == 0:
            print(f"⚠️ No common instances found between {baseline_alg} and {compare_alg}")
            continue
        
        # Create properly paired data
        baseline_paired_list = []
        compare_paired_list = []
        
        for instance_id in common_instances:
            # Get all runs for this instance
            baseline_runs = baseline_data[baseline_data['instance_id'] == instance_id]
            compare_runs = compare_data[compare_data['instance_id'] == instance_id]
            
            # Take minimum number of runs to ensure equal pairing
            min_runs = min(len(baseline_runs), len(compare_runs))
            
            if min_runs > 0:
                # Take first min_runs from each (or could randomize)
                baseline_subset = baseline_runs.head(min_runs)
                compare_subset = compare_runs.head(min_runs)
                
                baseline_paired_list.append(baseline_subset)
                compare_paired_list.append(compare_subset)
        
        if baseline_paired_list:
            # Combine all paired data
            baseline_paired = pd.concat(baseline_paired_list, ignore_index=True)
            compare_paired = pd.concat(compare_paired_list, ignore_index=True)
            
            # Ensure equal lengths
            min_length = min(len(baseline_paired), len(compare_paired))
            baseline_paired = baseline_paired.head(min_length)
            compare_paired = compare_paired.head(min_length)
            
            # Double check pairing
            assert len(baseline_paired) == len(compare_paired), f"Pairing failed: {len(baseline_paired)} vs {len(compare_paired)}"
            
            paired_results[compare_alg] = {
                'baseline': baseline_paired,
                'compare': compare_paired,
                'n_instances': len(common_instances),
                'n_pairs': len(baseline_paired)
            }
            
            print(f"✓ Created {len(baseline_paired)} paired runs from {len(common_instances)} instances for {baseline_alg} vs {compare_alg}")
    
    return paired_results

def perform_wilcoxon_tests(df, baseline_alg='GA', metrics=['total_cost', 'running_time'], alpha=0.05):
    """Perform Wilcoxon signed-rank tests comparing baseline algorithm with others"""
    print(f"\n🔬 WILCOXON SIGNED-RANK TEST: {baseline_alg} vs Other Algorithms")
    print("=" * 80)
    
    # Create paired dataset
    paired_data = create_paired_dataset(df, baseline_alg)
    if not paired_data:
        return None
    
    results = {}
    
    for metric in metrics:
        print(f"\n📊 METRIC: {metric.replace('_', ' ').title()}")
        print("-" * 60)
        
        metric_results = {}
        
        for compare_alg, data in paired_data.items():
            baseline_values = data['baseline'][metric].values
            compare_values = data['compare'][metric].values
            n_pairs = len(baseline_values)
            
            if n_pairs < 5:
                print(f"⚠️ {baseline_alg} vs {compare_alg}: Too few pairs ({n_pairs}) for reliable testing")
                continue
            
            # Descriptive statistics
            baseline_mean = np.mean(baseline_values)
            compare_mean = np.mean(compare_values)
            baseline_median = np.median(baseline_values)
            compare_median = np.median(compare_values)
            
            # Effect size (difference in means as percentage)
            if baseline_mean != 0:
                effect_size_pct = ((compare_mean - baseline_mean) / baseline_mean) * 100
            else:
                effect_size_pct = 0
            
            # Wilcoxon signed-rank test
            try:
                # Use wilcoxon test with zero_method='zsplit' to handle ties
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_values, compare_values, 
                                                          alternative='two-sided', 
                                                          zero_method='zsplit')
                
                # Additional statistics
                differences = compare_values - baseline_values
                n_positive = np.sum(differences > 0)  # Compare algorithm worse (higher values)
                n_negative = np.sum(differences < 0)  # Compare algorithm better (lower values)
                n_ties = np.sum(differences == 0)    # Tied results
                
                # Determine significance and winner
                is_significant = wilcoxon_p < alpha
                
                if is_significant:
                    if metric == 'total_cost':
                        # For total cost: LOWER is BETTER
                        if compare_mean < baseline_mean:
                            interpretation = f"✅ {compare_alg} is SIGNIFICANTLY BETTER than {baseline_alg} (lower cost)"
                            winner = compare_alg
                        else:
                            interpretation = f"✅ {baseline_alg} is SIGNIFICANTLY BETTER than {compare_alg} (lower cost)"
                            winner = baseline_alg
                    else:
                        # For running time: LOWER is BETTER
                        if compare_mean < baseline_mean:
                            interpretation = f"✅ {compare_alg} is SIGNIFICANTLY BETTER than {baseline_alg} (faster)"
                            winner = compare_alg
                        else:
                            interpretation = f"✅ {baseline_alg} is SIGNIFICANTLY BETTER than {compare_alg} (faster)"
                            winner = baseline_alg
                else:
                    interpretation = f"⚖️ NO SIGNIFICANT DIFFERENCE between {baseline_alg} and {compare_alg}"
                    winner = "Tie"
                
                # Store results
                metric_results[compare_alg] = {
                    'n_instances': data['n_instances'],
                    'n_pairs': n_pairs,
                    'baseline_mean': baseline_mean,
                    'compare_mean': compare_mean,
                    'baseline_median': baseline_median,
                    'compare_median': compare_median,
                    'effect_size_pct': effect_size_pct,
                    'wilcoxon_stat': wilcoxon_stat,
                    'wilcoxon_p': wilcoxon_p,
                    'is_significant': is_significant,
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'n_ties': n_ties,
                    'interpretation': interpretation,
                    'winner': winner
                }
                
                # Print detailed results
                print(f"\n🔸 {baseline_alg} vs {compare_alg}:")
                print(f"   Instances: {data['n_instances']} common instances")
                print(f"   Sample size: {n_pairs} paired runs")
                print(f"   {baseline_alg} mean: {baseline_mean:,.2f}")
                print(f"   {compare_alg} mean: {compare_mean:,.2f}")
                
                if metric == 'total_cost':
                    print(f"   Effect size: {effect_size_pct:+.2f}% ({'higher cost' if effect_size_pct > 0 else 'lower cost'} than {baseline_alg})")
                else:
                    print(f"   Effect size: {effect_size_pct:+.2f}% ({'slower' if effect_size_pct > 0 else 'faster'} than {baseline_alg})")
                
                print(f"   Wilcoxon statistic: {wilcoxon_stat}")
                print(f"   P-value: {wilcoxon_p:.6f}")
                print(f"   Significance (α={alpha}): {'YES' if is_significant else 'NO'}")
                print(f"   {compare_alg} better: {n_negative}/{n_pairs} runs ({n_negative/n_pairs*100:.1f}%)")
                print(f"   {baseline_alg} better: {n_positive}/{n_pairs} runs ({n_positive/n_pairs*100:.1f}%)")
                if n_ties > 0:
                    print(f"   Ties: {n_ties}/{n_pairs} runs ({n_ties/n_pairs*100:.1f}%)")
                print(f"   📋 CONCLUSION: {interpretation}")
                
            except Exception as e:
                print(f"❌ Wilcoxon test failed for {baseline_alg} vs {compare_alg}: {e}")
                continue
        
        results[metric] = metric_results
    
    return results

def print_summary(test_results, baseline_alg='GA'):
    """Print a summary of all test results"""
    if not test_results:
        print("❌ No test results to summarize")
        return
    
    print(f"\n📋 SUMMARY: {baseline_alg} Performance vs Competitors")
    print("=" * 80)
    
    for metric, metric_results in test_results.items():
        print(f"\n📊 {metric.replace('_', ' ').title()}:")
        
        significant_better = []
        significant_worse = []
        no_difference = []
        
        for compare_alg, result in metric_results.items():
            if result['is_significant']:
                if result['winner'] == baseline_alg:
                    significant_worse.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
                else:
                    significant_better.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
            else:
                no_difference.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
        
        if significant_better:
            print(f"   ✅ Algorithms SIGNIFICANTLY BETTER than {baseline_alg}:")
            for alg in significant_better:
                print(f"      • {alg}")
        
        if significant_worse:
            print(f"   ✅ {baseline_alg} SIGNIFICANTLY BETTER than:")
            for alg in significant_worse:
                print(f"      • {alg}")
        
        if no_difference:
            print(f"   ⚖️ NO SIGNIFICANT DIFFERENCE:")
            for alg in no_difference:
                print(f"      • {alg}")
        
        if not significant_better and not significant_worse and not no_difference:
            print(f"   ❓ No statistical comparisons available")

def save_results_to_file(test_results, baseline_alg='GA', output_dir='.'):
    """Save detailed test results to CSV and TXT files"""
    if not test_results:
        print("❌ No test results to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to CSV
    csv_filename = f"wilcoxon_results_{baseline_alg}_vs_others_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    detailed_results = []
    for metric, metric_results in test_results.items():
        for compare_alg, result in metric_results.items():
            detailed_results.append({
                'baseline_algorithm': baseline_alg,
                'compare_algorithm': compare_alg,
                'metric': metric,
                'n_instances': result['n_instances'],
                'n_pairs': result['n_pairs'],
                'baseline_mean': result['baseline_mean'],
                'compare_mean': result['compare_mean'],
                'baseline_median': result['baseline_median'],
                'compare_median': result['compare_median'],
                'effect_size_percent': result['effect_size_pct'],
                'wilcoxon_statistic': result['wilcoxon_stat'],
                'p_value': result['wilcoxon_p'],
                'is_significant': result['is_significant'],
                'winner': result['winner'],
                'interpretation': result['interpretation']
            })
    
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Detailed results saved to: {csv_filename}")
    
    # Save summary report to TXT
    txt_filename = f"wilcoxon_summary_{baseline_alg}_vs_others_{timestamp}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"WILCOXON SIGNED-RANK TEST RESULTS\n")
        f.write(f"Baseline Algorithm: {baseline_alg}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for metric, metric_results in test_results.items():
            f.write(f"METRIC: {metric.replace('_', ' ').title()}\n")
            f.write("-" * 60 + "\n")
            
            for compare_alg, result in metric_results.items():
                f.write(f"\n{baseline_alg} vs {compare_alg}:\n")
                f.write(f"   Instances: {result['n_instances']} common instances\n")
                f.write(f"   Sample size: {result['n_pairs']} paired runs\n")
                f.write(f"   {baseline_alg} mean: {result['baseline_mean']:,.2f}\n")
                f.write(f"   {compare_alg} mean: {result['compare_mean']:,.2f}\n")
                f.write(f"   Effect size: {result['effect_size_pct']:+.2f}%\n")
                f.write(f"   Wilcoxon statistic: {result['wilcoxon_stat']}\n")
                f.write(f"   P-value: {result['wilcoxon_p']:.6f}\n")
                f.write(f"   Significant (α=0.05): {'YES' if result['is_significant'] else 'NO'}\n")
                f.write(f"   Winner: {result['winner']}\n")
                f.write(f"   Conclusion: {result['interpretation']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        # Summary section
        f.write("\nSUMMARY BY METRIC:\n")
        f.write("=" * 40 + "\n")
        
        for metric, metric_results in test_results.items():
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            
            significant_better = []
            significant_worse = []
            no_difference = []
            
            for compare_alg, result in metric_results.items():
                if result['is_significant']:
                    if result['winner'] == baseline_alg:
                        significant_worse.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
                    else:
                        significant_better.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
                else:
                    no_difference.append(f"{compare_alg} (p={result['wilcoxon_p']:.4f})")
            
            if significant_better:
                f.write(f"   ✅ Algorithms SIGNIFICANTLY BETTER than {baseline_alg}:\n")
                for alg in significant_better:
                    f.write(f"      • {alg}\n")
            
            if significant_worse:
                f.write(f"   ✅ {baseline_alg} SIGNIFICANTLY BETTER than:\n")
                for alg in significant_worse:
                    f.write(f"      • {alg}\n")
            
            if no_difference:
                f.write(f"   ⚖️ NO SIGNIFICANT DIFFERENCE:\n")
                for alg in no_difference:
                    f.write(f"      • {alg}\n")
    
    print(f"✅ Summary report saved to: {txt_filename}")
    return csv_path, txt_path

def main():
    """Main function to run Wilcoxon tests only"""
    base_path = os.getcwd()
    
    print("🔬 Wilcoxon Signed-Rank Test for Algorithm Comparison")
    print("=" * 70)
    print(f"Looking for CSV files in: {base_path}/results/")
    print("Expected filename pattern: LOCATION_nc_X_ncl_X_nv_X_DATATYPE_r_X_X_X")
    print("-" * 70)
    
    # Load data
    df = load_data_for_testing(base_path)
    
    if df.empty:
        print("❌ No data loaded. Please check your file structure and naming convention.")
        return
    
    # Ask user which algorithm to use as baseline
    algorithms = sorted(df['algorithm'].unique())
    print(f"\nAvailable algorithms: {', '.join(algorithms)}")
    
    baseline_choice = input(f"Enter baseline algorithm for comparison (default: GA): ").strip()
    if not baseline_choice:
        baseline_choice = 'GA'
    elif baseline_choice.upper() not in [alg.upper() for alg in algorithms]:
        print(f"⚠️ '{baseline_choice}' not found. Using GA as default.")
        baseline_choice = 'GA'
    else:
        # Find exact match (case insensitive)
        baseline_choice = next(alg for alg in algorithms if alg.upper() == baseline_choice.upper())
    
    # Perform Wilcoxon tests
    test_results = perform_wilcoxon_tests(df, baseline_alg=baseline_choice, 
                                         metrics=['total_cost', 'running_time'], 
                                         alpha=0.05)
    
    # Print summary
    if test_results:
        print_summary(test_results, baseline_alg=baseline_choice)
        
        # Ask if user wants to save results
        save_choice = input(f"\nDo you want to save results to files? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            csv_file, txt_file = save_results_to_file(test_results, baseline_alg=baseline_choice)
            print(f"\n💾 Files saved:")
            print(f"   📊 Detailed data: {os.path.basename(csv_file)}")
            print(f"   📋 Summary report: {os.path.basename(txt_file)}")
    
    print(f"\n✅ Wilcoxon testing completed!")

if __name__ == "__main__":
    main()