import os
import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, distributed
from torchvision import models

from model.model_utils import get_dataloaders




"""
SHAMELESSLY copied and modified from the eecs-498-007 assignment code.
"""

class Solver(object):
    def __init__(self, model, batch_size, criterion, optimizer, config_dict, **kwargs):
        self.device = kwargs.pop("device", "cuda")
        self.max_data = kwargs.pop("max_data", None)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = criterion
        self.config_dict = config_dict
        dataloaders, dataset_sizes = get_dataloaders(batch_size, self.max_data, config_dict=config_dict)
        self.batch_size = batch_size
        self.val_dataloader = dataloaders["val"]
        self.train_dataloader = dataloaders["train"]
        self.dataset_sizes = dataset_sizes

        # Unpack keyword arguments
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", self.max_data if self.max_data else 100)
        self.num_val_samples = kwargs.pop("num_val_samples", self.max_data if self.max_data else 100)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.print_acc_every = kwargs.pop("print_acc_every", 1)
        self.verbose = kwargs.pop("verbose", True)
        self.data_dir = "/home/stoyelq/Documents/dfobot_data/"

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_acc_pm_history = []
        self.val_acc_pm_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data,
        images, labels = next(iter(self.train_dataloader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        output = self.model(images)
        loss = self.loss_function(output[:, 0], labels)
        self.optimizer.zero_grad()
        loss.backward()

        self.loss_history.append(loss.item())
        self.optimizer.step()


    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "batch_size": self.batch_size,
            "best_params": self.best_params,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
            "train_acc_pm_history": self.train_acc_pm_history,
            "val_acc_pm_history": self.val_acc_pm_history,
        }
        filename = "%s%s_epoch_%d.pkl" % (self.data_dir, self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, dataloader, num_samples, window=0):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - dataloader: Dataloader
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Compute predictions in batches
        self.model.eval()
        num_batches = num_samples // self.batch_size
        if num_samples % self.batch_size != 0:
            num_batches += 1
        y_pred = []
        y_true = []
        for i in range(num_batches):
            images, labels = next(iter(dataloader))
            images = images.to(self.device)
            labels = labels.to(self.device)

            scores = self.model(images)
            y_pred.append(scores[:, 0])
            y_true.append(labels)

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        acc = ((y_pred - y_true).abs().int() <= window).to(torch.float).mean()
        return acc.item()

    def train(self, return_best_params=True):
        """
        Run optimization to train the model.
        """
        num_train = self.dataset_sizes["train"]
        if self.max_data:
            num_train = self.max_data
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        prev_time = start_time = time.time()

        for t in range(num_iterations):
            cur_time = time.time()
            prev_time = cur_time
            self.model.train()
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Time %.2f sec; Iteration %d / %d) loss: %f"
                    % (
                        time.time() - start_time,
                        t + 1,
                        num_iterations,
                        self.loss_history[-1],
                    )
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            self.model.eval()
            with torch.no_grad():
                first_it = t == 0
                last_it = t == num_iterations - 1
                if first_it or last_it or epoch_end:
                    train_acc = \
                        self.check_accuracy(self.train_dataloader,
                                            num_samples=self.num_train_samples, window=0)
                    # train_acc_pm = \
                    #     self.check_accuracy(self.train_dataloader,
                    #                         num_samples=self.num_train_samples, window=1)
                    val_acc = \
                        self.check_accuracy(self.val_dataloader,
                                            num_samples=self.num_val_samples, window=0)
                    # val_acc_pm = \
                    #     self.check_accuracy(self.val_dataloader,
                    #                         num_samples=self.num_val_samples, window=1)
                    self.train_acc_history.append(train_acc)
                    # self.train_acc_pm_history.append(train_acc_pm)
                    self.val_acc_history.append(val_acc)
                    # self.val_acc_pm_history.append(val_acc_pm)
                    self._save_checkpoint()

                    if self.verbose and self.epoch % self.print_acc_every == 0:
                        print(
                            "(Epoch %d / %d) train acc: %f; val_acc: %f"
                            % (self.epoch, self.num_epochs, train_acc, val_acc)
                        )
                        # print(
                        #     "(Epoch %d / %d) train acc pm: %f; val_acc pm: %f"
                        #     % (self.epoch, self.num_epochs, train_acc_pm, val_acc_pm)
                        # )

                    # Keep track of the best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_params = self.model.state_dict()


        # At the end of training swap the best params into the model
        if return_best_params:
            self.model.state_dict = self.best_params


    def plot_solver_results(self, num_samples):
        num_batches = num_samples // self.batch_size
        if num_samples % self.batch_size != 0:
            num_batches += 1
        y_pred = []
        y_true = []
        best_model = self.model
        best_model.state_dict = self.best_params

        for i in range(num_batches):
            images, labels = next(iter(self.val_dataloader))
            images = images.to(self.device)
            labels = labels.to(self.device)
            scores = self.model(images)
            y_pred.append(torch.argmax(scores, dim=1))
            y_true.append(labels)

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        plt.scatter(y_true.tolist(), y_pred.tolist())
        plt.plot([0, 25], [0, 25])
        plt.show()
        return

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def run_solver(device, plots=False, all_layers=False, config_dict=None, save_count=None):
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    model_conv = models.resnet50(weights='IMAGENET1K_V2')
    model_conv.to(device)
    # freeze inner layers, if called for:
    for param in model_conv.parameters():
       param.requires_grad = all_layers

    # swap out final fc layer:
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 1) # go to single value

    criterion = nn.MSELoss()

    # all params:
    if all_layers:
        optimizer_ft = optim.Adam(model_conv.parameters(),
                                  lr=config_dict["LEARNING_RATE"],
                                  weight_decay=config_dict["WEIGHT_DECAY"])
    # just fc layer
    else:
        optimizer_ft = optim.Adam(model_conv.fc.parameters(),
                                  lr=config_dict["LEARNING_RATE"],
                                  weight_decay=config_dict["WEIGHT_DECAY"])

    solver = Solver(model_conv,
                    batch_size=config_dict["BATCH_SIZE"],
                    criterion=criterion,
                    optimizer=optimizer_ft,
                    config_dict=config_dict,
                    print_every=config_dict["PRINT_EVERY"],
                    num_epochs=config_dict["NUM_EPOCHS"],
                    max_data=config_dict["MAX_DATA"],
                    num_train_samples=config_dict["ACC_SAMPLES"],
                    num_val_samples=config_dict["ACC_VAL_SAMPLES"],
                    device=device,
                    )

    solver.train(return_best_params=False)
    if save_count:
        torch.save(solver, f"/home/stoyelq/Documents/dfobot_data/hyper_search/model_solver_{device.split(':')[-1]}_{save_count}.pt")
    else:
        torch.save(solver, f"/home/stoyelq/Documents/dfobot_data/model_solver_{device.split(':')[-1]}.pt")

    if plots:
        make_solver_plots(solver)
    else:
        make_solver_plots(solver, device=device, save_count=save_count)
    return solver


def make_solver_plots(solver, device=None, save_count=None):
    plt.plot(solver.loss_history, 'o')
    window_width = 20
    cumsum_vec = np.cumsum(np.insert(solver.loss_history, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(list(range(0, len(solver.loss_history) - window_width + 1)), ma_vec)
    if save_count:
        plt.savefig(f"/home/stoyelq/Documents/dfobot_data/hyper_search/loss_{device.split(':')[-1]}_{save_count}.png")
        plt.clf()
    else:
        plt.show()

    plt.plot(solver.train_acc_history, label="train_acc")
    plt.plot(solver.val_acc_history, label="train_acc_pm")
    plt.plot(solver.train_acc_pm_history, label="val_acc_pm")
    plt.plot(solver.val_acc_pm_history, label="val_acc")
    plt.title(f"Training and validation accuracy with lr: {solver.config_dict['LEARNING_RATE']}, weight decay: {solver.config_dict['WEIGHT_DECAY']}")
    if save_count:
        plt.savefig(f"/home/stoyelq/Documents/dfobot_data/hyper_search/acc_{device.split(':')[-1]}_{save_count}.png")
        plt.clf()
    else:
        plt.show()
    if save_count is None:
        solver.plot_solver_results(100)
