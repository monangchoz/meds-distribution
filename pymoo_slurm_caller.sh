#!/bin/bash

# List of algorithm names
algos=("ga","brkga","pso","de")  # <-- edit this list as needed

# Loop over each instance
for f in instances/*.json; do
    filename=$(basename "$f")
    
    # Loop over each algorithm name
    for algo in "${algos[@]}"; do
        sbatch --export=FILENAME=$filename,ALGONAME=$algo pymoo.slurm
    done
done
