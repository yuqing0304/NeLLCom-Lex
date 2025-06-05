# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from typing import List, Optional
import json

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq

        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)

        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    # def eval(self, data=None, args_print=None):
    #     mean_loss = 0.0
    #     interactions = []
    #     n_batches = 0
    #     validation_data = self.validation_data if data is None else data
    #     self.game.eval()
    #     if args_print is not None:
    #         epoch, id_to_color = args_print
    #         f = open(f"./msg_rf/output_{epoch}.txt", "w")
    #     with torch.no_grad():
    #         for batch in validation_data:
    #             if not isinstance(batch, Batch):
    #                 batch = Batch(*batch)
    #             batch = batch.to(self.device)
    #             optimized_loss, interaction = self.game(*batch)
    #             # print(f"dir of interaction: {dir(interaction)}")
    #             # print(sender_input, message, receiver_input, receiver_output, label)
    #             for i in range(interaction.sender_input.size(0)):
    #                 f.write(
    #                     f"{interaction.sender_input[i]} -> {id_to_color[interaction.message[i].argmax().item()]} -> {interaction.receiver_input[i]} -> {interaction.receiver_output[i].item()} -> {interaction.labels[i].item()}\n"
    #                 )
                    
    #             if (
    #                 self.distributed_context.is_distributed
    #                 and self.aggregate_interaction_logs
    #             ):
    #                 interaction = Interaction.gather_distributed_interactions(
    #                     interaction
    #                 )
    #             interaction = interaction.to("cpu")
    #             mean_loss += optimized_loss

    #             for callback in self.callbacks:
    #                 callback.on_batch_end(
    #                     interaction, optimized_loss, n_batches, is_training=False
    #                 )

    #             interactions.append(interaction)
    #             n_batches += 1
    #     f.close()

    #     mean_loss /= n_batches
    #     full_interaction = Interaction.from_iterable(interactions)

    #     return mean_loss.item(), full_interaction

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, interaction = self.game(*batch)

                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)

        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

    # # def train(self, n_epochs, id_to_color=None):
    # def train(self, n_epochs, id_to_color, current_seed):
    #     for callback in self.callbacks:
    #         callback.on_train_begin(self)

    #     for epoch in range(self.start_epoch, n_epochs):
    #         for callback in self.callbacks:
    #             callback.on_epoch_begin(epoch + 1)

    #         train_loss, train_interaction = self.train_epoch()

    #         for callback in self.callbacks:
    #             callback.on_epoch_end(train_loss, train_interaction, epoch + 1)

    #         validation_loss = validation_interaction = None
    #         if (
    #             self.validation_data is not None
    #             and self.validation_freq > 0
    #             and (epoch + 1) % self.validation_freq == 0
    #         ):
    #             for callback in self.callbacks:
    #                 callback.on_validation_begin(epoch + 1)
    #             # if id_to_color is not None:
    #             #     args_print = (epoch, id_to_color)
    #             #     validation_loss, validation_interaction = self.eval(args_print=args_print)

    #                 validation_loss, validation_interaction = self.eval(
    #                     data=self.validation_data,
    #                     args_print=(epoch, id_to_color),
    #                     seed=current_seed,  # Pass the current seed here
    #                 )

    #             for callback in self.callbacks:
    #                 callback.on_validation_end(
    #                     validation_loss, validation_interaction, epoch + 1
    #                 )

    #         if self.should_stop:
    #             for callback in self.callbacks:
    #                 callback.on_early_stopping(
    #                     train_loss,
    #                     train_interaction,
    #                     epoch + 1,
    #                     validation_loss,
    #                     validation_interaction,
    #                 )
    #             break

    #     for callback in self.callbacks:
    #         callback.on_train_end()


    def train(self, n_epochs, id_to_color, data_type, current_seed, if_context):

        base_dir = pathlib.Path(data_type)
        base_dir.mkdir(parents=True, exist_ok=True)

        if if_context:
            log_dir = base_dir / "training_log_context"
        else:
            log_dir = base_dir / "training_log"

        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_rf_seed{current_seed}.txt"

        with log_file.open("w") as log_f:
            for callback in self.callbacks:
                callback.on_train_begin(self)

    ### evaluation before training start ###
            # Initial evaluation before training starts
            if self.validation_data is not None:
                validation_loss, validation_interaction = self.eval(
                    data_type,
                    data=self.validation_data,
                    args_print=("Initial Eval", id_to_color),
                    seed=current_seed,
                    if_context=if_context,
                )
                # Convert tensors to scalars
                val_metrics = {
                    "loss": validation_loss,
                    "acc": validation_interaction.aux["acc"].float().mean().item(),
                    "acc_far": validation_interaction.aux["acc_far"].float().mean().item(),
                    "acc_close": validation_interaction.aux["acc_close"].float().mean().item(),
                    "acc_split": validation_interaction.aux["acc_split"].float().mean().item(),
                    "baseline": validation_interaction.aux["baseline"].float().mean().item(),
                    "sender_entropy": validation_interaction.aux["sender_entropy"].float().mean().item(),
                    "receiver_entropy": validation_interaction.aux["receiver_entropy"].float().mean().item(),
                    "mode": "test",
                    "epoch": 0,  # Indicate this is the initial evaluation
                }
                # Write initial evaluation metrics to log file
                log_f.write(f"{json.dumps(val_metrics)}\n")
    ### evaluation before training end ###


            for epoch in range(self.start_epoch, n_epochs):
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch + 1)

                # Train for one epoch
                train_loss, train_interaction = self.train_epoch()
                
                # Convert tensors to scalars
                train_metrics = {
                    "loss": train_loss,
                    "acc": train_interaction.aux["acc"].float().mean().item(),
                    "acc_far": train_interaction.aux["acc_far"].float().mean().item(),
                    "acc_close": train_interaction.aux["acc_close"].float().mean().item(),
                    "acc_split": train_interaction.aux["acc_split"].float().mean().item(),
                    "baseline": train_interaction.aux["baseline"].float().mean().item(),
                    "sender_entropy": train_interaction.aux["sender_entropy"].float().mean().item(),
                    "receiver_entropy": train_interaction.aux["receiver_entropy"].float().mean().item(),
                    "mode": "train",
                    "epoch": epoch + 1,
                }
                
                # Write metrics to log file in JSON format
                log_f.write(f"{json.dumps(train_metrics)}\n")

                for callback in self.callbacks:
                    callback.on_epoch_end(train_loss, train_interaction, epoch + 1)

                # Perform validation
                if (
                    self.validation_data is not None
                    and self.validation_freq > 0
                    and (epoch + 1) % self.validation_freq == 0
                ):

                    validation_loss, validation_interaction = self.eval(
                        data_type,
                        data=self.validation_data,
                        args_print=(epoch, id_to_color),
                        seed=current_seed,
                        if_context=if_context,  # Passing the if_context flag
                    )

                    # change epoch to batch level (author@zhu 2025-4-28)
                  
                    # Convert tensors to scalars
                    # print("===========type of interaction_aux=============")
                    # print(type(validation_interaction.aux["acc"]), validation_interaction.aux["acc"].shape)
                    # print(validation_interaction.aux["acc"])
                    val_metrics = {
                        "loss": validation_loss,
                        "acc": validation_interaction.aux["acc"].float().mean().item(),
                        "acc_check": ((validation_interaction.aux["acc"] == 1).sum().item(), len(validation_interaction.aux["acc"])),
                        # "acc_list": validation_interaction.aux["acc"].float().tolist(),
                        "acc_far": validation_interaction.aux["acc_far"].float().mean().item(),
                        "acc_far_check": ((validation_interaction.aux["acc_far"] == 1).sum().item(), len(validation_interaction.aux["acc_far"])),
                        # "acc_far_list": validation_interaction.aux["acc_far"].float().tolist(),
                        "acc_close": validation_interaction.aux["acc_close"].float().mean().item(),
                        "acc_close_check": ((validation_interaction.aux["acc_close"] == 1).sum().item(), len(validation_interaction.aux["acc_close"])),
                        # "acc_close_list": validation_interaction.aux["acc_close"].float().tolist(),
                        "acc_split": validation_interaction.aux["acc_split"].float().mean().item(),
                        "acc_split_check": ((validation_interaction.aux["acc_split"] == 1).sum().item(), len(validation_interaction.aux["acc_split"])),
                        # "acc_split_list": validation_interaction.aux["acc_split"].float().tolist(),
                        "baseline": validation_interaction.aux["baseline"].float().mean().item(),
                        "sender_entropy": validation_interaction.aux["sender_entropy"].float().mean().item(),
                        "receiver_entropy": validation_interaction.aux["receiver_entropy"].float().mean().item(),
                        "mode": "test",
                        "epoch": epoch + 1,
                    }
                    
                    # Write validation metrics to log file in JSON format
                    log_f.write(f"{json.dumps(val_metrics)}\n")

                    for callback in self.callbacks:
                        callback.on_validation_end(
                            validation_loss, validation_interaction, epoch + 1
                        )
                
                # Check if the training should stop
                if self.should_stop:
                    for callback in self.callbacks:
                        callback.on_early_stopping(
                            train_loss,
                            train_interaction,
                            epoch + 1,
                            validation_loss,
                            validation_interaction,
                        )
                    break
            
            # End of training
            for callback in self.callbacks:
                callback.on_train_end()



    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)


    def eval(self, data_type, data=None, args_print=None, seed=None, if_context=False):
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        validation_data = self.validation_data if data is None else data
        self.game.eval()

        base_dir = pathlib.Path(data_type)
        base_dir.mkdir(parents=True, exist_ok=True)

        if if_context:
            dump_dir = base_dir / "dump_context"
        else:
            dump_dir = base_dir / "dump"

        dump_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            output_dir = dump_dir / f"msg_rf_seed{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"output_{args_print[0]}.txt" if args_print else output_dir / "output.txt"
        else:
            output_dir = dump_dir / "msg_rf"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "output.txt"

        ## N/A
        # # Define the directory for the seed
        # if seed is not None:
        #     output_dir = pathlib.Path(f"./msg_rf_seed{seed}")
        #     output_dir.mkdir(parents=True, exist_ok=True)
        #     output_file = output_dir / f"output_{args_print[0]}.txt" if args_print else output_dir / "output.txt"
        # else:
        #     output_file = pathlib.Path(f"./msg_rf/output.txt")

        with output_file.open("w") as f:
            with torch.no_grad():
                for batch in validation_data:
                    if not isinstance(batch, Batch):
                        batch = Batch(*batch)
                    batch = batch.to(self.device)
                    optimized_loss, interaction = self.game(*batch)

                    for i in range(interaction.sender_input.size(0)):
                        # add condition to dump the output @zhu 2025-02-25
                        condition = interaction.aux_input['condition'][i]
                        if condition == 0:
                            c_name = 'far'
                        elif condition == 1:
                            c_name = 'close'
                        elif condition == 2:
                            c_name = 'split'
                        else:
                            print(f"condition: {condition}")
                            raise ValueError("Invalid condition value")
                        f.write(
                            f"{[[int(x) for x in sublist] for sublist in interaction.sender_input[i].tolist()]} -> "
                            f"{args_print[1][interaction.message[i].item()]} ({args_print[1][interaction.aux_input['color'][i].item()]}) -> "
                            f"{[[int(x) for x in sublist] for sublist in interaction.receiver_input[i].tolist()]} -> "
                            f"{interaction.receiver_output[i].item()} -> "
                            f"{interaction.labels[i].item()} -> "
                            f"{c_name}\n"
                        )
                        
                        # f.write(
                        #     f"{interaction.sender_input[i].tolist()} -> "
                        #     f"{args_print[1][interaction.message[i].item()]} -> "
                        #     f"{interaction.receiver_input[i].tolist()} -> "
                        #     f"{interaction.receiver_output[i].item()} -> "
                        #     f"{interaction.labels[i].item()}\n"
                        # )
                        # if interaction.receiver_output[i].item() != interaction.labels[i].item():
                        #     f.write(
                        #         f"{interaction.sender_input[i]} -> "
                        #         f"{args_print[1][interaction.message[i].item()]} -> "
                        #         f"{interaction.receiver_input[i]} -> "
                        #         f"{interaction.receiver_output[i].item()} -> "
                        #         f"{interaction.labels[i].item()}\n"
                        #     )                             
                        # f.write(f"{interaction.message[i].item()}\n")

                        # f.write(
                        #     f"{interaction.sender_input[i]} -> "
                        #     f"{args_print[1][interaction.message[i].argmax().item()]} -> "
                        #     f"{interaction.receiver_input[i]} -> "
                        #     f"{interaction.receiver_output[i].item()} -> "
                        #     f"{interaction.labels[i].item()}\n"
                        # )
                        # f.write(
                        #     f"{interaction.sender_input[i]} -> {id_to_color[interaction.message[i].argmax().item()]} -> {interaction.receiver_input[i]} -> {interaction.receiver_output[i].item()} -> {interaction.labels[i].item()}\n"
                        # )                        
                    if (
                        self.distributed_context.is_distributed
                        and self.aggregate_interaction_logs
                    ):
                        interaction = Interaction.gather_distributed_interactions(
                            interaction
                        )
                    interaction = interaction.to("cpu")
                    mean_loss += optimized_loss

                    for callback in self.callbacks:
                        callback.on_batch_end(
                            interaction, optimized_loss, n_batches, is_training=False
                        )

                    interactions.append(interaction)
                    n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction
