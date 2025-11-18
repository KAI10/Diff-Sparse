#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -A <account_name>
#SBATCH -p <partition_name>
#SBATCH --constraint=a100_80gb
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -J dcrnn
#SBATCH -o dcrnn-%j.out
#SBATCH -e dcrnn-%j.err


module load miniforge
conda activate genai_env

#python train.py
python train.py --config config_$1.json
