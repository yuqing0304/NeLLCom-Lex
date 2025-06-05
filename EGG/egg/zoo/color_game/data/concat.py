import csv

condition_slrl_path = 'condition_allhuman_cielab.csv'
sample_far_path = './same_base_data/sample_far.txt'
sample_split_path = './same_base_data/sample_split.txt'
sample_close_path = './same_base_data/sample_close.txt'

with open(sample_far_path, 'r') as far_file:
    far_lines = far_file.readlines()

with open(sample_split_path, 'r') as split_file:
    split_lines = split_file.readlines()

with open(sample_close_path, 'r') as close_file:
    close_lines = close_file.readlines()


############# experiment 1 ################

# condition_slrl_path = 'condition_allhuman_cielab.csv'
# sample_far_path = './different_base_data/sample_far.txt'
# sample_split_path = './different_base_data/sample_split.txt'
# sample_close_path = './different_base_data/sample_close.txt'

# new_path = "condition_slrl2.csv"
# lines_to_append = far_lines[:16808] + split_lines[:7017] + close_lines[:4043] 
# Far: 16,808, Split: 7,017, Close: 4,043. --- condition_slrl2 (double, the same ratio)  # 7,499, 3,131, and 1,804

# new_path = "condition_slrlhuman.csv" # --- eval on the same 3000 data for both SL and RL
# lines_to_append = close_lines[:12434] 
############# experiment 1 end ################


############# experiment 2 ################

half_eval_len = int(15434/2)
half_train_len = int(15434/2)
train_len = int(15434)
skip_len = int(15434/2)

new_path = "condition_a.csv"
lines_to_append = far_lines[:half_eval_len] + close_lines[:half_eval_len + train_len] # 0/100

new_path = "condition_b.csv"
lines_to_append = far_lines[:half_eval_len + half_train_len] + close_lines[:half_eval_len] + close_lines[half_eval_len + skip_len:half_eval_len + skip_len + half_train_len] # 50/50

new_path = "condition_c.csv"
lines_to_append = far_lines[:half_eval_len + train_len] + close_lines[:half_eval_len] # 100/0


############# experiment 2 end ################


# Read original lines
with open(condition_slrl_path, 'r', encoding='utf-8') as f:
    original_lines = f.readlines()

# Clean up new lines
cleaned_lines_to_append = [line.strip() + "\n" for line in lines_to_append]

# Combine both
combined_lines = original_lines + ["\n"] + cleaned_lines_to_append

# Write to a new file
with open(new_path, 'w', encoding='utf-8') as f:
    f.writelines(combined_lines)

