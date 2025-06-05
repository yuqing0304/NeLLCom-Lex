import os
import json
import csv
import ast
import re
import random
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



def save_new_csv():
    data_path = './reordered_output.csv'

    # get all the permutations of 0, 1, 2
    options = list(itertools.permutations([0, 1, 2]))
    # convert tuples to lists
    options = [list(item) for item in options]

    # read csv
    df = pd.read_csv(data_path, index_col=None)

    # new csv
    output_file = './new_output.csv'
    f_w = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f_w)
    for idx, row in df.iterrows():
        tar_HLS = ast.literal_eval(row['tar_HLS'])  
        dist1_HLS = ast.literal_eval(row['dist1_HLS'])
        dist2_HLS = ast.literal_eval(row['dist2_HLS'])
        label = row['contents_cleaned']

        
        position = random.choice(options)
        # print(f"position: {position}")

        if position == [0, 1, 2]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, tar_HLS, dist1_HLS, dist2_HLS, 0, label]
        elif position == [0, 2, 1]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, tar_HLS, dist2_HLS, dist1_HLS, 0, label]
        elif position == [1, 0, 2]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, dist1_HLS, tar_HLS, dist2_HLS, 1, label]
        elif position == [1, 2, 0]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, dist2_HLS, tar_HLS, dist1_HLS, 2, label]
        elif position == [2, 0, 1]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, dist1_HLS, dist2_HLS, tar_HLS, 1, label]
        elif position == [2, 1, 0]:
            row = [tar_HLS, dist1_HLS, dist2_HLS, dist2_HLS, dist1_HLS, tar_HLS, 2, label]
        else:
            print("Error")
        writer.writerow(row)  
    f_w.close()


def read_new_csv():
    data_path = './new_output.csv'
    # read csv
    df = pd.read_csv(data_path, index_col=None)
    # get data and labels
    for idx, row in df.iterrows():
        fixed_list1, fixed_list2, fixed_list3, list1, list2, list3, position, label = row
        print(f"list1: {list1}")
        print(f"list2: {list2}")
        print(f"list3: {list3}")
        print(f"position: {position}")
        print(f"label: {label}")
        break


if __name__ == "__main__":
    save_new_csv()