#!/bin/bash

# List of algorithm names
declare -a algos=("brkga")  # <-- edit this list as needed

# Loop over each instance
for f in instances/*.json; do
    filename=$(basename "$f")
    
    # Loop over each algorithm name
    for algo in "${algos[@]}"; do
        echo "Submitting $filename with $algo"
        sbatch --export=FILENAME=$filename,ALGONAME=$algo pymoo.slurm
    done
done

