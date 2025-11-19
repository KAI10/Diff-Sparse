#!/bin/bash
#SBATCH -A <account_name>
#SBATCH -p <partition_name>
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 06:00:00

module load miniforge
conda activate <environment_name>

echo Random Seed: "$seed"
echo Store path: "$storePath"

python3.12 lightning_training.py --seed=$seed --storePath=$storePath
