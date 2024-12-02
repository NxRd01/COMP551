#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551
#SBATCH -e %N-%j.err # STDERR

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

python a4_baselinemodels.py
