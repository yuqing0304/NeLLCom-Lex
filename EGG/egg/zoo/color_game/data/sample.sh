#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:20:00
#SBATCH --output=out.out


source activate comm

# python sample_cielab.py
python sample_cielab_samebase.py