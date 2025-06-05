import os
import torch
import torch.nn as nn
import pathlib
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from egg.zoo.color_gamezero.models import NewReceiver, NewSender
from egg.zoo.color_game.datasets import UniformConditionDataset
import numpy as np
import random


def cal_accuracy(output, labels):
    pred = output.argmax(dim=1)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    return correct / len(labels)



def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_data(data_path):

    sl_data_set = UniformConditionDataset(data_path, rf=False)
    rf_data_set = UniformConditionDataset(data_path, rf=True)

    # ############# original begin (sl+rl: both human data) ###############
    # half_len = len(sl_data_set) // 2
    # first_dataset = Subset(sl_data_set, range(half_len))
    # second_dataset = Subset(rf_data_set, range(half_len, len(rf_data_set)))


    # sl_train_size = int(0.8 * len(first_dataset))
    # sl_valid_size = len(first_dataset) - sl_train_size
    # print("sl_train_size", sl_train_size)
    # print("sl_valid_size", sl_valid_size)

    # rf_train_size = int(0.8 * len(second_dataset))
    # rf_valid_size = len(second_dataset) - rf_train_size
    # print("rf_train_size", rf_train_size)
    # print("rf_valid_size", rf_valid_size)

    # sl_train_set,sl_valid_set = random_split(first_dataset, [sl_train_size, sl_valid_size])
    # rf_train_set,rf_valid_set = random_split(second_dataset, [rf_train_size, rf_valid_size])
    # ############# original end ###############


    # ############ for qualitative analysis: rl train generated + rl eval human 3000 begin ###############
    """
    # condition 2: test =identical set of samples for humans and agents => fair comparison, but small
    # far:split:close = 9309:3886:2239, 15434 in total--- 12434 for train; 3000 for eval
    # 15434 - 3000 = 12434 for rl train generated: 7,499, 3,131, and 1,804 --- 12434 generated for train, the same 3000 for eval, 27868 in total
    """
    half_len = (len(sl_data_set) -3000) // 2 + 3000
    first_dataset = Subset(sl_data_set, range(half_len))
    second_dataset = Subset(rf_data_set, range(half_len, len(rf_data_set)))

    sl_train_size = len(first_dataset) - 3000  
    sl_valid_size = 3000  #

    generator = torch.Generator().manual_seed(111) 

    sl_train_set, sl_valid_set = random_split(first_dataset, [sl_train_size, sl_valid_size], generator=generator)
    rf_train_set = second_dataset

    sl_valid_indices = sl_valid_set.indices  

    rf_valid_set = Subset(rf_data_set, sl_valid_indices)
    # ############ rl train generated + rl eval human 3000 end ###############


    # # ################ data split for far/split/close begin ################ 
    # """ 
    # condition3: all-human-data vs. generated-data (of same size) => no direct comparison possible, but large. Only compute system-level metrics here.
    # # 15434 - 3000 = 12434 for rl train generated: 7,499, 3,131, and 1,804 --- 12434 (7,499, 3,131, and 1,804) generated for train, 15434 (9309:3886:2239) generated for eval, so in total generate Far: 16,808, Split: 7,017, Close: 4,043. 
    # # 43,302 in total
    # """
    # sl_len = 15434
    # first_dataset = Subset(sl_data_set, range(sl_len))
    # second_dataset = Subset(rf_data_set, range(sl_len, len(rf_data_set)))
    # sl_train_size = len(first_dataset) - 3000  
    # sl_valid_size = 3000 
    # generator = torch.Generator().manual_seed(111)  
    # sl_train_set, sl_valid_set = random_split(first_dataset, [sl_train_size, sl_valid_size], generator=generator)


############ exp2: three distributions begin #############
    # far_valid_size = int(15434/2)
    # close_valid_size = int(15434/2)


    # if "condition_a" in data_path:
    #     # 0/100 setting: condition_a
    #     rf_far_len = int(15434/2 + 0)
    #     rf_split_len = 0
    #     rf_close_len = int(15434/2 + 15434)
    # elif "condition_b" in data_path:
    #     # 50/50 setting: condition_b
    #     rf_far_len = int(15434/2 + 15434/2)
    #     rf_split_len = 0
    #     rf_close_len = int(15434/2 + 15434/2)
    # elif "condition_c" in data_path:
    #     # 100/0 setting: condition_c
    #     rf_far_len = int(15434/2 + 15434)
    #     rf_split_len = 0
    #     rf_close_len = int(15434/2 + 0)


    # far_dataset = Subset(second_dataset, range(0, rf_far_len))
    # # split_dataset = Subset(second_dataset, range(rf_far_len, rf_far_len + rf_split_len))
    # close_dataset = Subset(second_dataset, range(rf_far_len + rf_split_len, rf_far_len + rf_split_len + rf_close_len))

    # far_valid_set = Subset(far_dataset, range(far_valid_size))
    # far_train_set = Subset(far_dataset, range(far_valid_size, len(far_dataset)))

    # close_valid_set = Subset(close_dataset, range(close_valid_size))
    # close_train_set = Subset(close_dataset, range(close_valid_size, len(close_dataset)))

    # rf_train_set = ConcatDataset([far_train_set, close_train_set])
    # rf_valid_set = ConcatDataset([far_valid_set, close_valid_set])
############ three distributions end #############


# ##############  exp1 #####################
    # if "condition_slrl2" in data_path:
    #     # original setting: condition_slrl2
    #     rf_far_len = 16808
    #     rf_split_len = 7017
    #     rf_close_len = 4043 

    # if "condition_slrl2" in data_path:
    #     far_train_size = 7499
    #     split_train_size = 3131
    #     close_train_size = 1804

    # far_dataset = Subset(second_dataset, range(0, rf_far_len))
    # split_dataset = Subset(second_dataset, range(rf_far_len, rf_far_len + rf_split_len))
    # close_dataset = Subset(second_dataset, range(rf_far_len + rf_split_len, rf_far_len + rf_split_len + rf_close_len))

    # far_train_set = Subset(far_dataset, range(far_train_size))
    # far_valid_set = Subset(far_dataset, range(far_train_size, len(far_dataset)))

    # split_train_set = Subset(split_dataset, range(split_train_size))
    # split_valid_set = Subset(split_dataset, range(split_train_size, len(split_dataset)))

    # close_train_set = Subset(close_dataset, range(close_train_size))
    # close_valid_set = Subset(close_dataset, range(close_train_size, len(close_dataset)))

    # # Combine the far, split, and close datasets for training and validation
    # rf_train_set = ConcatDataset([far_train_set, split_train_set, close_train_set])
    # rf_valid_set = ConcatDataset([far_valid_set, split_valid_set, close_valid_set])
# ##############  exp1 end #####################

    # # # ################ data split for far/split/close end ################

    sl_train_loader = DataLoader(sl_train_set, batch_size=32, shuffle=True, drop_last=False, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(111), num_workers=0)
    sl_valid_loader = DataLoader(sl_valid_set, batch_size=32, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(111), num_workers=0)
    rf_train_loader = DataLoader(rf_train_set, batch_size=32, shuffle=True, drop_last=False, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(111))
    rf_valid_loader = DataLoader(rf_valid_set, batch_size=32, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(111))

    return sl_train_loader, sl_valid_loader, rf_train_loader, rf_valid_loader, rf_data_set.color_num, rf_data_set.id_to_color



def create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device): # num_classes = color_num
    model = NewSender(embed_dim, hidden_dim, color_num, dropout, if_context).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device):
    model = NewReceiver(embed_dim, hidden_dim, color_num, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def train_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, data_type, seed):

    base_dir = pathlib.Path(data_type)
    base_dir.mkdir(parents=True, exist_ok=True)

    if if_context:
        training_log_dir = base_dir / "training_log_context"
        dump_dir = base_dir / "dump_context"
    else:
        training_log_dir = base_dir / "training_log"
        dump_dir = base_dir / "dump"

    training_log_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = training_log_dir / f"log_spk_seed{seed}.txt"  # Log file for accuracy
    spk_folder = dump_dir / f"msg_spk_seed{seed}"
    os.makedirs(spk_folder, exist_ok=True)

    model, optimizer = create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device)
    
    with open(log_file_path, "w") as log_file:  # Open log file for writing
        for epoch in range(n_epochs):
            model.train()
            train_outs = []
            train_ids = []
            ### three conditions begin 
            far_outs, far_ids = [], []
            close_outs, close_ids = [], []
            split_outs, split_ids = [], []
            ### three conditions end
            for batch_idx, (sender_input, position, receiver_input, color_ids, conditions) in enumerate(train_loader):
                sender_input = sender_input.to(device)
                color_ids = color_ids.to(device)
                optimizer.zero_grad()
                output = model(sender_input)
                loss = nn.CrossEntropyLoss()(output, color_ids)
                loss.backward()
                optimizer.step()
                train_outs.append(output)
                train_ids.append(color_ids)
                # print(condition)
                ### three conditions begin
                for condition in conditions:
                    if condition == 'far':
                        far_outs.append(output)
                        far_ids.append(color_ids)
                    elif condition == 'close':
                        close_outs.append(output)
                        close_ids.append(color_ids)
                    elif condition == 'split':
                        split_outs.append(output)
                        split_ids.append(color_ids)
                    else:
                        print('Condition Error!!!')
                ### three conditions end

                # # evaluate for each batch
                # model.eval()
                # with torch.no_grad():
                #     valid_outs = []
                #     valid_ids = []
                #     ### three conditions begin
                #     far_outs, far_ids = [], []
                #     close_outs, close_ids = [], []
                #     split_outs, split_ids = [], []
                #     ### three conditions end
                #     # with open(os.path.join(spk_folder, f"sender_epoch{epoch}.txt"), "w") as f:
                #     for i, (sender_input, position, receiver_input, color_ids, conditions) in enumerate(valid_loader):
                #         sender_input = sender_input.to(device)
                #         color_ids = color_ids.to(device)
                #         output = model(sender_input)
                #         valid_outs.append(output)
                #         valid_ids.append(color_ids)
                #         # three conditions begin
                #         for condition in conditions:
                #             if condition == 'far':
                #                 far_outs.append(output)
                #                 far_ids.append(color_ids)
                #             elif condition == 'close':
                #                 close_outs.append(output)
                #                 close_ids.append(color_ids)
                #             elif condition == 'split':
                #                 split_outs.append(output)
                #                 split_ids.append(color_ids)
                #             else:
                #                 print('Condition Error!!!')
                #         # three conditions end
                #         colors = [id_to_color[color_id.item()] for color_id in color_ids]
                #         preds = [id_to_color[output[i].argmax().item()] for i in range(len(output))]
                #             # for i in range(sender_input.shape[0]):
                #                 # f.write(f"{[[int(x) for x in sublist] for sublist in sender_input[i].tolist()]} -> {preds[i]} (label={colors[i]}) -> {conditions[i]}\n")
                #     acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_ids))
                #     ### three conditions begin
                #     acc_far = cal_accuracy(torch.cat(far_outs), torch.cat(far_ids))
                #     acc_close = cal_accuracy(torch.cat(close_outs), torch.cat(close_ids))
                #     acc_split = cal_accuracy(torch.cat(split_outs), torch.cat(split_ids))
                #     ### three conditions end
                    
                #     # log_file.write(f"Epoch {epoch}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split} \n")  # Log validation accuracy
                #     # print for each 50 batches
                #     if batch_idx % 50 == 0:
                #         print(f"Epoch {epoch} batch: {batch_idx}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}")

            acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_ids))

            ### three conditions begin
            acc_far = cal_accuracy(torch.cat(far_outs), torch.cat(far_ids))
            acc_close = cal_accuracy(torch.cat(close_outs), torch.cat(close_ids))
            acc_split = cal_accuracy(torch.cat(split_outs), torch.cat(split_ids))
            ### three conditions end
            log_file.write(f"Epoch {epoch}, train acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split} \n")  # Log training accuracy
            print(f"Epoch {epoch}, train acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}")
            
            model.eval()
            with torch.no_grad():
                valid_outs = []
                valid_ids = []
                ### three conditions begin
                far_outs, far_ids = [], []
                close_outs, close_ids = [], []
                split_outs, split_ids = [], []
                ### three conditions end
                with open(os.path.join(spk_folder, f"sender_epoch{epoch}.txt"), "w") as f:
                    for i, (sender_input, position, receiver_input, color_ids, conditions) in enumerate(valid_loader):
                        sender_input = sender_input.to(device)
                        color_ids = color_ids.to(device)
                        output = model(sender_input)
                        valid_outs.append(output)
                        valid_ids.append(color_ids)
                        # three conditions begin
                        for condition in conditions:
                            if condition == 'far':
                                far_outs.append(output)
                                far_ids.append(color_ids)
                            elif condition == 'close':
                                close_outs.append(output)
                                close_ids.append(color_ids)
                            elif condition == 'split':
                                split_outs.append(output)
                                split_ids.append(color_ids)
                            else:
                                print('Condition Error!!!')
                        # three conditions end

                        colors = [id_to_color[color_id.item()] for color_id in color_ids]
                        preds = [id_to_color[output[i].argmax().item()] for i in range(len(output))]
                        for i in range(sender_input.shape[0]):
                            f.write(f"{[[int(x) for x in sublist] for sublist in sender_input[i].tolist()]} -> {preds[i]} (label={colors[i]}) -> {conditions[i]}\n")
                
                acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_ids))

                ### three conditions begin
                acc_far = cal_accuracy(torch.cat(far_outs), torch.cat(far_ids))
                acc_close = cal_accuracy(torch.cat(close_outs), torch.cat(close_ids))
                acc_split = cal_accuracy(torch.cat(split_outs), torch.cat(split_ids))
                ### three conditions end
                
                log_file.write(f"Epoch {epoch}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split} \n")  # Log validation accuracy
                print(f"Epoch {epoch}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}")
    return model


def train_receiver(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, data_type, seed):

    base_dir = pathlib.Path(data_type)
    base_dir.mkdir(parents=True, exist_ok=True)

    if if_context:
        training_log_dir = base_dir / "training_log_context"
        dump_dir = base_dir / "dump_context"
    else:
        training_log_dir = base_dir / "training_log"
        dump_dir = base_dir / "dump"

    training_log_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = training_log_dir / f"log_lst_seed{seed}.txt"  # Log file for accuracy
    lst_folder = dump_dir / f"msg_lst_seed{seed}"
    os.makedirs(lst_folder, exist_ok=True)
        
    model, optimizer = create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device)
    
    with open(log_file_path, "w") as log_file:  # Open log file for writing
        for epoch in range(n_epochs):
            model.train()
            train_outs = []
            train_pos = []
            
            ### three conditions begin
            far_outs, far_pos = [], []
            close_outs, close_pos = [], []
            split_outs, split_pos = [], []
            ### three conditions end

            for i, (sender_input, position, receiver_input, color_ids, conditions) in enumerate(train_loader):
                receiver_input = receiver_input.to(device)
                color_ids = color_ids.to(device)
                position = position.to(device)
                optimizer.zero_grad()
                output = model(color_ids, receiver_input)
                loss = nn.CrossEntropyLoss()(output, position)
                loss.backward()
                optimizer.step()
                train_outs.append(output)
                train_pos.append(position)
                
                ### three conditions begin
                for condition in conditions:
                    if condition == 'far':
                        far_outs.append(output)
                        far_pos.append(position)
                    elif condition == 'close':
                        close_outs.append(output)
                        close_pos.append(position)
                    elif condition == 'split':
                        split_outs.append(output)
                        split_pos.append(position)
                    else:
                        print('Condition Error!!!')
                ### three conditions end

            acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_pos))            
            acc_far = cal_accuracy(torch.cat(far_outs), torch.cat(far_pos))
            # print(f"far_pos, {len(far_pos)}")
            acc_close = cal_accuracy(torch.cat(close_outs), torch.cat(close_pos))
            # print(f"close_pos, {len(close_pos)}")            
            acc_split = cal_accuracy(torch.cat(split_outs), torch.cat(split_pos))
            # print(f"split_pos, {len(split_pos)}")                       
            log_file.write(f"Epoch {epoch}, train acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}\n")
            print(f"Epoch {epoch}, train acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}")

            model.eval()
            with torch.no_grad():
                valid_outs = []
                valid_pos = []
                
                ### three conditions begin
                far_outs, far_pos = [], []
                close_outs, close_pos = [], []
                split_outs, split_pos = [], []
                ### three conditions end
                
                with open(os.path.join(lst_folder, f"receiver_epoch{epoch}.txt"), "w") as f:
                    for i, (sender_input, position, receiver_input, color_ids, conditions) in enumerate(valid_loader):
                        receiver_input = receiver_input.to(device)
                        color_ids = color_ids.to(device)
                        position = position.to(device)
                        output = model(color_ids, receiver_input)
                        valid_outs.append(output)
                        valid_pos.append(position)
                        
                        ### three conditions begin
                        for condition in conditions:
                            if condition == 'far':
                                far_outs.append(output)
                                far_pos.append(position)
                            elif condition == 'close':
                                close_outs.append(output)
                                close_pos.append(position)
                            elif condition == 'split':
                                split_outs.append(output)
                                split_pos.append(position)
                            else:
                                print('Condition Error!!!')
                        ### three conditions end

                        preds = output.argmax(dim=1)
                        colors = [id_to_color[color_id.item()] for color_id in color_ids]
                        for i in range(receiver_input.shape[0]):
                            f.write(
                                f"{colors[i]} -> {[[int(x) for x in sublist] for sublist in receiver_input[i].tolist()]} -> {preds[i]} (label={position[i]})\n"
                            )

                acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_pos))
                acc_far = cal_accuracy(torch.cat(far_outs), torch.cat(far_pos))
                # print(f"far_pos, {len(far_pos)}")
                acc_close = cal_accuracy(torch.cat(close_outs), torch.cat(close_pos))
                # print(f"close_pos, {len(close_pos)}")
                acc_split = cal_accuracy(torch.cat(split_outs), torch.cat(split_pos))
                # print(f"split_pos, {len(split_pos)}")                
                
                log_file.write(f"Epoch {epoch}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}\n")
                print(f"Epoch {epoch}, valid acc: {acc}, far_acc: {acc_far}, close_acc: {acc_close}, split_acc: {acc_split}")

    return model
