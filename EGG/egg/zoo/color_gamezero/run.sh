#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:30:00
#SBATCH --output=out.out


source activate comm

# python train.py --data_path ../color_game/data/condition_a.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ../color_game/data/condition_b.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ../color_game/data/condition_c.csv --n_epochs 30 --n_comm_epochs 30 

python train.py --data_path ../color_game/data/different_base_data/condition_slrl2.csv --n_epochs 30 --n_comm_epochs 30 
