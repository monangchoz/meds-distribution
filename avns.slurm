#!/bin/bash
#SBATCH --job-name=avns
#SBATCH --output=logs/avns_%j.out
#SBATCH --error=logs/avns_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1

source /home/mahasiswa1/miniconda3/bin/activate

python main_avns.py --instance-file-name "$FILENAME" --max-iteration 100 --patience 10