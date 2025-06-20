#!/bin/bash
#SBATCH -A nssac_students
#SBATCH -p bii-gpu
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 10:00:00

module load miniforge
conda activate pytorch-venv

echo Random Seed: "$seed"
echo Store path: "$storePath"

python3.12 lightning_training.py --seed=$seed --storePath=$storePath
