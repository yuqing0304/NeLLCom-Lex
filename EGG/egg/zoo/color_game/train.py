# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import random
import argparse
import json
import torch
import os
import egg.core as core
from egg.zoo.color_game.utils_condition import load_data, train_receiver, train_sender

current_dir = os.path.dirname(os.path.abspath(__file__))

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training configurations
    parser.add_argument("--mode", type=str, default="rf", choices=["rf", "gs"],
                        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.")
    parser.add_argument("--spk_hidden_dim", type=int, default=512, help="Hidden layer dimension.") 
    parser.add_argument("--lst_hidden_dim", type=int, default=512, help="Hidden layer dimension") 
    # parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension.") 
    parser.add_argument("--embed_dim", type=int, default=3, help="Input embedding dimension. Default: 3.")
    parser.add_argument("--gs_tau", type=float, default=1.0, help="Gumbel-Softmax temperature. Default: 1.0.")
    # parser.add_argument("--entropy_coeff", type=float, default=0.01, 
    #                     help="Entropy regularization coefficient for Reinforce mode. Default: 0.01.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default: 0.2.")
    parser.add_argument("--if_context", action="store_true", help="Set context for sender. Default: False.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs. Default: 10.")
    parser.add_argument("--n_comm_epochs", type=int, default=20, help="Number of communication training epochs. Default: 10.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate. Default: 0.01.") 
    parser.add_argument("--comm_lr", type=float, default=0.00001, help="Communication learning rate. Default: 0.01.") 
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default: 32.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file.")
                          
    opt = core.init(parser)
    return opt



def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()

    condition = _aux_input["condition"]

    acc_far = acc[condition == 0]
    acc_close = acc[condition == 1]
    acc_split = acc[condition == 2]

    return -acc, {"acc": acc, "acc_far": acc_far, "acc_close": acc_close, "acc_split": acc_split}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL (negative log-likelihood) loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}



def get_game(opts, sender, receiver):
    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.15,
            receiver_entropy_coeff=0.15,
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game


def sl_train(color_num, embed_dim, spk_hidden_dim, lst_hidden_dim, dropout, if_context, lr, train_loader, valid_loader, n_epochs, id_to_color, data_type, seed):
    '''
        Supervised learning training for sender and receiver
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Receiver...")
    receiver = train_receiver(color_num, embed_dim, lst_hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, data_type, seed) 
    print("Successfully trained Receiver!")
    print("Training Sender...")
    sender = train_sender(color_num, embed_dim, spk_hidden_dim,  dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, data_type, seed)
    print("Successfully trained Sender!")
    return sender, receiver



def main():
    opts = parse_arguments()
    color_num = 3

    def generate_random_seeds(n=10, seed_range=(0, 10000)):
        return random.sample(range(*seed_range), n)

    seed_list = generate_random_seeds()
    print(seed_list)

    for seed in seed_list:
        print(f"Running experiment with seed: {seed}")
        set_seed(seed)  

        print("Loading data...")
        data_path = opts.data_path
        data_type = os.path.splitext(os.path.basename(data_path))[0]

        sl_train_loader, sl_valid_loader, rf_train_loader, rf_valid_loader, color_num, id_to_color = load_data(data_path)
        print("Start Supervised Learning...")
        print(f'if_context: {opts.if_context}')
        
        print("Successfully trained Sender and Receiver!")
        print("*" * 15)
        print("Start Reinforcement Learning...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training Receiver...")
        receiver = train_receiver(color_num, opts.embed_dim, opts.spk_hidden_dim, opts.dropout, opts.if_context, opts.lr, sl_train_loader, sl_valid_loader, device, opts.n_epochs, id_to_color, data_type, seed)
        print("Training Sender...")
        sender = train_sender(color_num, opts.embed_dim, opts.spk_hidden_dim, opts.dropout, opts.if_context, opts.lr, sl_train_loader, sl_valid_loader, device, opts.n_epochs, id_to_color, data_type, seed)
        game = get_game(opts, sender, receiver)
        '''original code for optimizer'''
        optimizer = core.build_optimizer(game.parameters(), rate=opts.comm_lr)
        '''freezed receiver for optimizer'''
        # # freeze receiver parameters
        # for param in receiver.parameters():
        #     param.requires_grad = False
        # update sender's parameters only
        # optimizer = core.build_optimizer(filter(lambda p: p.requires_grad, game.parameters()), rate=opts.comm_lr)


        callbacks = []
        if opts.mode == "gs":
            callbacks.append(core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1))
        
        callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))


        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=rf_train_loader,
            validation_data=rf_valid_loader,
            callbacks=callbacks,
        )

        trainer.train(n_epochs=opts.n_comm_epochs, id_to_color=id_to_color, data_type = data_type, current_seed=seed, if_context=opts.if_context)

        print(f"Training with seed {seed} completed.")
    
    print("All experiments completed.")
    core.close()


if __name__ == "__main__":
    main()

