import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
import json
from matplotlib.ticker import FuncFormatter

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

def load_all_instance_files_directly(base_path):
    """
    Load ALL instance files directly from instances folder
    Count every single JSON file, even if they have similar names but different content
    """
    instances_path = os.path.join(base_path, 'instances')
    
    if not os.path.exists(instances_path):
        print(f"Instances folder not found at: {instances_path}")
        return pd.DataFrame()
    
    # Get all JSON files from instances folder
    json_files = glob.glob(os.path.join(instances_path, "*.json"))
    
    if not json_files:
        # Try other possible extensions
        other_extensions = ["*.txt", "*.dat"]
        for ext in other_extensions:
            json_files.extend(glob.glob(os.path.join(instances_path, ext)))
    
    print(f"🔍 Found {len(json_files)} files in instances folder")
    
    all_instances = []
    processed_count = 0
    
    for instance_file_path in json_files:
        filename = os.path.basename(instance_file_path)
        
        # Read reefer information
        reefer_info = read_instance_file(instance_file_path)
        
        if reefer_info is not None:
            # Extract metadata from filename
            location, num_customers, num_clusters, num_vehicles, data_type, ratios = extract_detailed_info(filename)
            
            instance_record = {
                'filename': filename,
                'location': location,
                'num_customers': num_customers,
                'num_clusters': num_clusters,
                'num_vehicles': num_vehicles,
                'data_type': data_type,
                'ratio_small': ratios[0] if ratios else None,
                'ratio_medium': ratios[1] if ratios else None,
                'ratio_large': ratios[2] if ratios else None,
                'reefer_percent': reefer_info['reefer_percent'],
                'non_reefer_percent': reefer_info['non_reefer_percent'],
                'reefer_count': reefer_info['reefer_count'],
                'total_items_instance': reefer_info['total_items'],
                'total_customers_instance': reefer_info['total_customers'],
                'file_path': instance_file_path
            }
            
            all_instances.append(instance_record)
            processed_count += 1
            
            if processed_count % 50 == 0:  # Progress indicator
                print(f"   Processed {processed_count} instances...")
            
            # Detailed logging for first few and some random instances
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"✓ {filename}: {location}, nc:{num_customers}, ncl:{num_clusters}, {data_type} ({reefer_info['reefer_percent']:.1f}% reefer)")
        
        else:
            print(f"✗ Failed to process: {filename}")
    
    print(f"\n📊 Successfully processed {processed_count} out of {len(json_files)} files")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_instances)
    
    return df

def check_instance_folder_contents(base_path):
    """
    Check what's actually in the instances folder
    """
    instances_path = os.path.join(base_path, 'instances')
    
    if not os.path.exists(instances_path):
        print(f"❌ Instances folder not found at: {instances_path}")
        return
    
    print(f"\n🔍 CHECKING INSTANCES FOLDER CONTENTS:")
    print(f"📁 Path: {instances_path}")
    print("-" * 60)
    
    # Count files by extension
    all_files = os.listdir(instances_path)
    file_extensions = {}
    
    for file in all_files:
        if os.path.isfile(os.path.join(instances_path, file)):
            _, ext = os.path.splitext(file)
            ext = ext.lower() if ext else 'no_extension'
            file_extensions[ext] = file_extensions.get(ext, 0) + 1
    
    print(f"📊 File count by extension:")
    for ext, count in sorted(file_extensions.items()):
        print(f"   {ext}: {count} files")
    
    print(f"\n📋 Total files in folder: {len(all_files)}")
    
    # Show first few filenames as examples
    print(f"\n📝 First 10 filenames (examples):")
    for i, filename in enumerate(sorted(all_files)[:10]):
        print(f"   {i+1:2d}. {filename}")
    
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more files")

def create_simple_reefer_histogram(df):
    """
    Create single histogram showing distribution of problem instances by reefer percentage
    X-axis: Reefer percentage (%)
    Y-axis: Number of instances
    """
    if df.empty:
        print("⚠️ No data available for histogram")
        return None
    
    # Get reefer percentages
    reefer_percentages = df['reefer_percent'].dropna()
    
    if reefer_percentages.empty:
        print("⚠️ No valid reefer data found for histogram")
        return None
    
    print(f"📊 Creating histogram for {len(reefer_percentages)} problem instances")
    
    # Create single histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create histogram with appropriate bins
    n_bins = min(30, len(reefer_percentages.unique()))  # Adaptive number of bins
    counts, bins, patches = ax.hist(reefer_percentages, bins=n_bins, alpha=0.7, 
                                   color='steelblue', edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_title('Distribusi Problem Instance berdasarkan Persentase Barang Berpendingin', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Persentase Barang Berpendingin (%)', fontsize=14)
    ax.set_ylabel('Jumlah Problem Instance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars (only if not too many bars)
    if len(counts) <= 20:
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                height = patch.get_height()
                ax.text(patch.get_x() + patch.get_width()/2., height + max(counts)*0.01,
                       f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def print_detailed_summary(df):
    """
    Print detailed summary of problem instances
    """
    if df.empty:
        print("⚠️ No data available for summary")
        return
    
    reefer_percentages = df['reefer_percent'].dropna()
    
    print("\n" + "=" * 70)
    print("📊 RINGKASAN DISTRIBUSI PROBLEM INSTANCE")
    print("=" * 70)
    
    print(f"\n📈 STATISTIK UMUM:")
    print(f"   Total problem instance: {len(reefer_percentages)}")
    print(f"   Rentang persentase reefer: {reefer_percentages.min():.1f}% - {reefer_percentages.max():.1f}%")
    print(f"   Rata-rata persentase reefer: {reefer_percentages.mean():.1f}%")
    print(f"   Median persentase reefer: {reefer_percentages.median():.1f}%")
    print(f"   Standar deviasi: {reefer_percentages.std():.1f}%")
    
    # Show unique reefer percentages
    unique_reefer = sorted(reefer_percentages.unique())
    print(f"   Nilai reefer unik: {len(unique_reefer)} nilai berbeda")
    print(f"   Rentang nilai: {unique_reefer[:5]}...{unique_reefer[-5:] if len(unique_reefer) > 10 else unique_reefer[5:]}")
    
    # Frequency distribution (show top 10 most common)
    print(f"\n🎯 10 PERSENTASE REEFER PALING UMUM:")
    print("-" * 45)
    reefer_counts = df['reefer_percent'].value_counts().head(10)
    
    for reefer_pct, count in reefer_counts.items():
        percentage_of_total = (count / len(reefer_percentages)) * 100
        print(f"   {reefer_pct:5.1f}% reefer: {count:3d} instance ({percentage_of_total:4.1f}% dari total)")
    
    # Range-based analysis
    print(f"\n📊 ANALISIS BERDASARKAN RENTANG:")
    print("-" * 40)
    ranges = [
        (0, 10, "Sangat Rendah (0-10%)"),
        (10, 25, "Rendah (10-25%)"),
        (25, 50, "Sedang (25-50%)"),
        (50, 75, "Tinggi (50-75%)"),
        (75, 100, "Sangat Tinggi (75-100%)")
    ]
    
    for min_val, max_val, label in ranges:
        if min_val == 75:  # Include 100% in the last range
            count = len(df[df['reefer_percent'] >= min_val])
        else:
            count = len(df[
                (df['reefer_percent'] >= min_val) & 
                (df['reefer_percent'] < max_val)
            ])
        
        percentage = (count / len(reefer_percentages)) * 100 if len(reefer_percentages) > 0 else 0
        print(f"   {label:25s}: {count:3d} instance ({percentage:4.1f}%)")
    
    # Data composition
    print(f"\n📋 KOMPOSISI DATA:")
    print("-" * 25)
    
    # By location
    if 'location' in df.columns:
        locations = df['location'].dropna().value_counts()
        if not locations.empty:
            print(f"   Berdasarkan Lokasi Depot:")
            for location, count in locations.items():
                percentage = (count / len(df)) * 100
                print(f"      {location}: {count} instance ({percentage:.1f}%)")
    
    # By data type
    if 'data_type' in df.columns:
        data_types = df['data_type'].dropna().value_counts()
        if not data_types.empty:
            print(f"   Berdasarkan Tipe Data:")
            for data_type, count in data_types.items():
                percentage = (count / len(df)) * 100
                print(f"      {data_type.title()}: {count} instance ({percentage:.1f}%)")

def main():
    """
    Main function for complete reefer distribution analysis
    """
    base_path = os.getcwd()
    
    print("🧊 ANALISIS LENGKAP DISTRIBUSI PERSENTASE BARANG BERPENDINGIN")
    print("=" * 70)
    print("Script ini menganalisis SEMUA file instance yang ada di folder instances")
    print("(menghitung setiap file JSON terpisah, termasuk yang serupa tapi berbeda isi)")
    print("=" * 70)
    
    # First, check what's in the instances folder
    check_instance_folder_contents(base_path)
    
    # Load ALL instance files directly
    print("\n🔍 Memuat SEMUA file instance dari folder instances...")
    df = load_all_instance_files_directly(base_path)
    
    if df.empty:
        print("❌ Tidak ditemukan file instance dengan data reefer.")
        print("\nPastikan:")
        print("  1. Folder 'instances' ada di direktori ini")
        print("  2. File instance berformat JSON dengan field 'is_reefer_required'")
        print("  3. Ada file .json di dalam folder instances")
        return
    
    print(f"\n📊 TOTAL FILE INSTANCE YANG BERHASIL DIPROSES: {len(df)}")
    
    # Create simple histogram
    print("\n📈 Membuat histogram distribusi reefer...")
    fig = create_simple_reefer_histogram(df)
    
    if fig:
        plt.show()
        
        # Print detailed summary
        print_detailed_summary(df)
        
        # Save option
        save_plot = input("\nApakah Anda ingin menyimpan histogram? (y/n): ").lower().strip()
        
        if save_plot == 'y':
            filename = 'distribusi_reefer_semua_instance.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Histogram disimpan sebagai {filename}")
        
        # Export data option
        export_data = input("\nApakah Anda ingin mengekspor data ke CSV? (y/n): ").lower().strip()
        
        if export_data == 'y':
            csv_filename = 'semua_instance_reefer_data.csv'
            df.to_csv(csv_filename, index=False)
            print(f"✅ Data diekspor ke {csv_filename}")
        
        print("\n🎉 Analisis distribusi reefer lengkap selesai!")
    
    else:
        print("❌ Tidak dapat membuat histogram. Periksa data reefer Anda.")

if __name__ == "__main__":
    main()