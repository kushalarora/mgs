#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dong.qian@mila.quebec
#SBATCH -o ./log/slurm-%j.out

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate torch181

python eval_generation.py --model_path=$1
