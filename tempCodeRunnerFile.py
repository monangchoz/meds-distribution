# ===== 4. PRINT RESULTS =====
def print_results ():
    for i, (center, cluster) in enumerate(zip(cluster_centers, clusters)):
        print(f"Cluster {i+1}: Center = {center}")
        print(f"Points ({len(cluster)}):")
        print(cluster)
        print(f"Max distance from center: {max([haversine(center, p, unit='km') for p in cluster]):.2f} km\n")
