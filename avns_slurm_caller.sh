#!/bin/bash

for f in instances/*.json; do
    filename=$(basename "$f")
    sbatch --export=FILENAME=$filename avns.slurm
done