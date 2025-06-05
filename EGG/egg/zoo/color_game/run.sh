#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --output=out.out


source activate comm

############ experiment1 ##################
# python train.py --data_path ./data/different_base_data/condition_slrl2.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/different_base_data/condition_slrl2.csv --n_epochs 1 --n_comm_epochs 30 


############ experiment2 ##################
# python train.py --data_path ./data/condition_a.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_a.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ./data/condition_c.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_c.csv --n_epochs 30 --n_comm_epochs 30 

############ qualitative eval experiment ##################
python train.py --data_path ./data/condition_slrlhuman.csv --n_epochs 30 --n_comm_epochs 30 --if_context

# mv ./training_log ./training_log_context ./dump ./dump_context ../color_gamezero
