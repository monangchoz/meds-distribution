#!/bin/bash
#SBATCH --job-name=avns
#SBATCH --output=logs/avns_%j.out
#SBATCH --error=logs/avns_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1

source /home/mahasiswa1/miniconda3/bin/activate

python main_pymoo.py --instance-file-name "$FILENAME" --algo-name "$ALGONAME" --patience 30