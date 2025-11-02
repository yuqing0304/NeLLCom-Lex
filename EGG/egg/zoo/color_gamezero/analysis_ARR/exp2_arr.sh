#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --partition=staging
#SBATCH --output=out.out


source activate comm




################# plot1: accuracy over epochs (3 conditions) ###############
# python exp2_commacc.py \
#   --log_folder1 ../condition3_generated_ARR/experiment2/condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ../condition3_generated_ARR/experiment2/condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ../condition3_generated_ARR/experiment2/condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc \
#   --output_plot accuracy_plot_overall.pdf

# python exp2_commacc.py \
#   --log_folder1 ../condition3_generated_ARR/experiment2/condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ../condition3_generated_ARR/experiment2/condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ../condition3_generated_ARR/experiment2/condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc_far \
#   --output_plot accuracy_plot_far.pdf

# python exp2_commacc.py \
#   --log_folder1 ../condition3_generated_ARR/experiment2/condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ../condition3_generated_ARR/experiment2/condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ../condition3_generated_ARR/experiment2/condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc_close \
#   --output_plot accuracy_plot_close.pdf




################# sec2: informativeness ###############
################# repeat this for condition a, b, c ###############
# seed_list=(111 222 333 444 555 666 777 888 999 123 234 345 456 567 678 789 891 912)

# for seed in "${seed_list[@]}"; do
#   input_file="../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed${seed}/output_epoch30.txt"
#   output_file="../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed${seed}/epoch30.txt"

#   if [ -f "$input_file" ]; then
#     python 4_informativeness.py \
#       --input_files "$input_file" \
#       --output_file "$output_file" \
#       --rl_or_sl >> ../condition3_generated_ARR/experiment2/condition_b/dump_context/informativeness.txt
#   else
#     echo "Skipping seed ${seed}: file not found"
#   fi
# done


# for seed in "${seed_list[@]}"; do
#   input_file="../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed${seed}/output_epoch0.txt"
#   output_file="../condition3_generated_ARR/experiment2/condition_b/dump_context/msg_rf_seed${seed}/epoch0.txt"

#   if [ -f "$input_file" ]; then
#     python 4_informativeness.py \
#       --input_files "$input_file" \
#       --output_file "$output_file" \
#       --rl_or_sl >> ../condition3_generated_ARR/experiment2/condition_b/dump_context/spk_informativeness.txt
#   else
#     echo "Skipping seed ${seed}: file not found"
#   fi
# done




################# sec3: system-level informativeness (3 conditions) ###############
python exp2_systeminfo.py 




################# sec6: entropy of informativeness ###############
python exp2_entropyinfo.py



################# sec4: CIELAB plots ###############
# python exp2_cielab.py --base_path ../condition3_generated_ARR/experiment2/condition_b/dump_context/ --out_dir ./denotations 




################# sec5: word count plot (3 conditions) ###############
python exp2_wordcount.py 






