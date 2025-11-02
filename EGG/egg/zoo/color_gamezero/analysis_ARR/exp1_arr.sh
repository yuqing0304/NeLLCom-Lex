#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --partition=staging
#SBATCH --output=out.out


source activate comm



python 1_plot_sl.py --log_folder1 ../condition3_generated_ARR/experiment1/training_log --log_prefix1 log_lst \
--log_folder2 ../condition3_generated_ARR/experiment1/training_log_context --log_prefix2 log_lst \
--log_folder3 ../condition3_generated_ARR/experiment1/training_log_exp --log_prefix3 log_lst \
--accuracy_type valid --output_plot acc_lst_sl_generated.pdf


python 1_plot_sl.py --log_folder1 ../condition3_generated_ARR/experiment1/training_log --log_prefix1 log_spk \
--log_folder2 ../condition3_generated_ARR/experiment1/training_log_context --log_prefix2 log_spk \
--log_folder3 ../condition3_generated_ARR/experiment1/training_log_exp --log_prefix3 log_spk \
--accuracy_type valid --output_plot acc_spk_sl_generated.pdf

python 1_plot_rf.py --log_folder1 ../condition3_generated_ARR/experiment1/training_log --log_prefix1 log_rf \
--log_folder2 ../condition3_generated_ARR/experiment1/training_log_context --log_prefix2 log_rf \
--log_folder3 ../condition3_generated_ARR/experiment1/training_log_exp --log_prefix3 log_rf \
--accuracy_type valid --output_plot acc_comm_rf_generated.pdf





################# sec2: informativeness ###############
################### after RL ################### 

# seed_list=(111 123 222 333 345 456 567 777 891 999)

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump/informativeness.txt
# done

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump_context/informativeness.txt
# done

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump_exp/informativeness.txt
# done

################### after SL ################### 


# seed_list=(111 123 222 333 345 456 567 777 891 999)

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump/msg_rf_seed${seed}/output_epoch0.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump/msg_rf_seed${seed}/epoch0.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump/spk_informativeness.txt
# done

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed${seed}/output_epoch0.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed${seed}/epoch0.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump_context/spk_informativeness.txt
# done

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed${seed}/output_epoch0.txt \
#     --output_file ../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed${seed}/epoch0.txt \
#     --rl_or_sl >> ../condition3_generated_ARR/experiment1/dump_exp/spk_informativeness.txt
# done





################# sec3: prototypes ###############
################ prototypes ###############
# epochs=$(seq 0 30)       # Epochs 0 through 30

# for seed in "${seed_list[@]}"; do
#     for epoch in $epochs; do
#         python 2_prototype.py ../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

#         python 2_prototype.py ../condition3_generated_ARR/experiment1/dump/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

#         python 2_prototype.py ../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl
#     done
# done




################# sec drift ###############
# python 2_avg_drift.py \
#   --epoch 30 \
#   --seeds 111 123 222 333 345 456 567 777 891 999 \
#   --label_prototype ./prototypes.pkl \
#   --base_dirs \
#     dump=../condition3_generated_ARR/experiment1/dump/msg_rf_seed{seed}/ \
#     dump_exp=../condition3_generated_ARR/experiment1/dump_exp/msg_rf_seed{seed}/ \
#     dump_context=../condition3_generated_ARR/experiment1/dump_context/msg_rf_seed{seed}/ \
#   --last_epoch_num 10 \
#   --output drift_epoch30_summary.csv


python 2_avg_drift.py \
  --epoch 30 \
  --seeds 111 123 222 333 345 456 567 777 891 999 \
  --label_prototype ./prototypes.pkl \
  --base_dirs \
    dump=../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed{seed}/ \
    dump_exp=../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed{seed}/ \
    dump_context=../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed{seed}/ \
  --last_epoch_num 30 \
  --output drift_epoch30_summary.csv

