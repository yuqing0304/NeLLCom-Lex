#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=staging
#SBATCH --output=out.out


source activate comm



# python 1_plot_rf.py --log_folder1 ../condition3_generated_ARR/exp_group/condition_b_2lst/training_log_context --log_prefix1 log_rf \
# --log_folder2 ../condition3_generated_ARR/exp_group/condition_b_4lst/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition3_generated_ARR/exp_group/condition_b_6lst/training_log_context --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_sl_generated.pdf




################# sec2: informativeness ###############
################### after RL ################### 
# seed_list=(111 123 222 333 345 456 567 777 891 999)
# seed_list=(111 222 333 444 555 666 777 888 999)
seed_list=(111 123 333 345 444 555 567 666 777 912)
# seed_list=(111)
lst_list=(1lst 2lst 6lst 10lst)
total_epochs=30

for lst in "${lst_list[@]}"; do
  # extract just the number part (e.g., "2" from "2lst")
  n=${lst%lst}
  max_epoch=$(( total_epochs / n ))

  for epoch_num in 0 $max_epoch; do
    for seed in "${seed_list[@]}"; do
      python 4_informativeness.py \
        --input_files ../../color_game_group/condition_bupsample100_${lst}/dump_context/msg_rf_seed${seed}/output_epoch${epoch_num}.txt \
        --output_file ../../color_game_group/condition_bupsample100_${lst}/dump_context/msg_rf_seed${seed}/epoch${epoch_num}.txt \
        --rl_or_sl >> ../../color_game_group/condition_bupsample100_${lst}/dump_context/informativeness.txt
    done
  done
done






# ################# sec3: prototypes ###############
# seed_list=(111 123 333 345 444 555 567 666 777 912)
seed_list=(111)
# lst_list=(1lst 2lst 6lst 10lst)
lst_list=(1lst)

total_epochs=30

for lst in "${lst_list[@]}"; do
  # extract just the number part (e.g., "2" from "2lst")
  n=${lst%lst}
  epoch_num=$(( total_epochs / n ))  # integer division

  # make sure it's treated as an integer
  for seed in "${seed_list[@]}"; do
      for epoch in $(seq 0 $epoch_num); do
          python 2_prototype.py "../../color_game_group/condition_bupsample100_${lst}/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt" \
              --pickle_output "prototypes_rf_human_epoch${epoch}.pkl" \
              --csv_output "prototypes_rf_human_epoch${epoch}.csv" --rl
      done
  done
done







################# sec drift ###############
seed_list=(111 123 333 345 444 555 567 666 777 912)
# lst_list=(1lst 2lst 6lst 10lst)
lst_list=(1lst)
total_epochs=30

for lst in "${lst_list[@]}"; do
  # extract just the number part (e.g., "2" from "2lst")
  n=${lst%lst}
  last_epoch_n=$(( total_epochs / n ))

  python 2_avg_drift.py \
    --epoch $last_epoch_n \
    --seeds "${seed_list[@]}" \
    --label_prototype ./prototypes.pkl \
    --base_dirs \
      dump_context=../../color_game_group/condition_bupsample0_${lst}/dump_context/msg_rf_seed{seed}/ \
    --last_epoch_num $last_epoch_n \
    --output drift_condition_bupsample0_${lst}.csv
done

