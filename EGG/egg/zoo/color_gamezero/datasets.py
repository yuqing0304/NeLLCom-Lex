import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast


class UniformDataset(Dataset):
    def __init__(self, data_path, rf=True):
        self.rf = rf
        self.sender_input, self.receiver_input, self.positions, self.color_ids, self.id_to_color = self._load_data(data_path)
        self.color_num = len(set(self.color_ids))

    def _load_data(self, data_path):
        # read csv
        df = pd.read_csv(data_path, index_col=None)
        # get data and labels
        sender_input = []
        receiver_input = []
        positions = []
        colors = []
        for idx, row in df.iterrows():
            fixed_list1, fixed_list2, fixed_list3, list1, list2, list3, position, label = row
            fixed_list1 = ast.literal_eval(fixed_list1)
            fixed_list2 = ast.literal_eval(fixed_list2)
            fixed_list3 = ast.literal_eval(fixed_list3)
            list1 = ast.literal_eval(list1)
            list2 = ast.literal_eval(list2)
            list3 = ast.literal_eval(list3)
            sender_input.append([fixed_list1, fixed_list2, fixed_list3])
            receiver_input.append([list1, list2, list3])
            positions.append(position)
            colors.append(label)
        colors_to_id = {color: idx for idx, color in enumerate(sorted(set(colors)))}
        ids_to_colors = {idx: color for color, idx in colors_to_id.items()}
        print(f"id_to_colors: {ids_to_colors}")
        color_ids = [colors_to_id[color] for color in colors]
        return sender_input, receiver_input, positions, color_ids, ids_to_colors
    
    def __len__(self):
        return len(self.sender_input)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        sender_input = torch.stack([torch.tensor(self.sender_input[idx][i]).float() for i in range(len(self.sender_input[idx]))])
        receiver_input = torch.stack([torch.tensor(self.receiver_input[idx][i]).float() for i in range(len(self.receiver_input[idx]))])
        if self.rf:
            return sender_input, position, receiver_input
        else:
            return sender_input, position, receiver_input, np.array(self.color_ids[idx])




class UniformConditionDataset(Dataset):
    def __init__(self, data_path, rf=True):
        self.rf = rf
        self.sender_input, self.receiver_input, self.positions, self.color_ids, self.id_to_color, self.conditions = self._load_data(data_path)
        self.color_num = len(set(self.color_ids))

    def _load_data(self, data_path):
        df = pd.read_csv(data_path, index_col=None)
        sender_input = []
        receiver_input = []
        positions = []
        colors = []
        conditions = []
        for idx, row in df.iterrows():
            fixed_list1, fixed_list2, fixed_list3, list1, list2, list3, position, label, condition = row
            fixed_list1 = ast.literal_eval(fixed_list1)
            fixed_list2 = ast.literal_eval(fixed_list2)
            fixed_list3 = ast.literal_eval(fixed_list3)
            list1 = ast.literal_eval(list1)
            list2 = ast.literal_eval(list2)
            list3 = ast.literal_eval(list3)
            sender_input.append([fixed_list1, fixed_list2, fixed_list3])
            receiver_input.append([list1, list2, list3])
            positions.append(position)
            colors.append(label)
            conditions.append(condition)
        colors_to_id = {color: idx for idx, color in enumerate(sorted(set(colors)))}
        ids_to_colors = {idx: color for color, idx in colors_to_id.items()}
        print(f"id_to_colors: {ids_to_colors}")
        color_ids = [colors_to_id[color] for color in colors]
        return sender_input, receiver_input, positions, color_ids, ids_to_colors, conditions
    
    def __len__(self):
        return len(self.sender_input)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        sender_input = torch.stack([torch.tensor(self.sender_input[idx][i]).float() for i in range(len(self.sender_input[idx]))])
        receiver_input = torch.stack([torch.tensor(self.receiver_input[idx][i]).float() for i in range(len(self.receiver_input[idx]))])
        condition = self.conditions[idx]
        aux_input = {}
        if condition == 'far':
            aux_input['condition'] = 0
        elif condition == 'close':
            aux_input['condition'] = 1
        elif condition == 'split':
            aux_input['condition'] = 2
        else:
            print("Unknown condition")
        if self.rf:
            return sender_input, position, receiver_input, aux_input
        else:
            return sender_input, position, receiver_input, np.array(self.color_ids[idx]), condition