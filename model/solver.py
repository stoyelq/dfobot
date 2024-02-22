import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import models

from model.model_utils import get_dataloaders


LEARNING_RATE = 0.005
BATCH_SIZE = 10
PRINT_EVERY = 10
MAX_DATA = None # 150
NUM_EPOCHS = 10
LR_DECAY = None #0.5
WEIGHT_DECAY = 0 #1e-4
ACC_SAMPLES = 100
ACC_VAL_SAMPLES = 1


"""
SHAMELESSLY copied and modified from the eecs-498-007 assignment code.
"""

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules.
    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    Example usage might look something like this:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
            lr_decay=0.95,
            num_epochs=10, batch_size=100,
            print_every=100,
            device='cuda')
    solver.train()
    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to torch
      tensors containing parameter values.
    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].
      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].
      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    def __init__(self, model, batch_size, criterion, optimizer, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        Optional arguments:
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - print_acc_every: We will print the accuracy every
          print_acc_every epochs.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.device = kwargs.pop("device", "cuda")
        self.max_data = kwargs.pop("max_data", None)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = criterion
        dataloaders, dataset_sizes, class_names = get_dataloaders(batch_size, self.max_data)
        self.batch_size = batch_size
        self.val_dataloader = dataloaders["val"]
        self.train_dataloader = dataloaders["train"]
        self.class_names = class_names
        self.dataset_sizes = dataset_sizes

        # Unpack keyword arguments
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
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
        loss = self.loss_function(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()


    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "lr_decay": self.lr_decay,
            "batch_size": self.batch_size,
            "best_params": self.best_params,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s%s_epoch_%d.pkl" % (self.data_dir, self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, dataloader, num_samples=None):
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
            y_pred.append(torch.argmax(scores, dim=1))
            y_true.append(labels)

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        acc = (y_pred == y_true).to(torch.float).mean()
        return acc.item()

    def train(self, time_limit=None, return_best_params=True):
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
            if (time_limit is not None) and (t > 0):
                next_time = cur_time - prev_time
                if cur_time - start_time + next_time > time_limit:
                    print(
                        "(Time %.2f sec; Iteration %d / %d) loss: %f"
                        % (
                            cur_time - start_time,
                            t,
                            num_iterations,
                            self.loss_history[-1],
                        )
                    )
                    print("End of training; next iteration "
                          "will exceed the time limit.")
                    break
            prev_time = cur_time

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
            with torch.no_grad():
                first_it = t == 0
                last_it = t == num_iterations - 1
                if first_it:
                    print(f"Learning rate: {LEARNING_RATE}")
                    print(f"Batch size: {self.batch_size}")

                if first_it or last_it or epoch_end:
                    train_acc = \
                        self.check_accuracy(self.train_dataloader,
                                            num_samples=self.num_train_samples)
                    val_acc = \
                        self.check_accuracy(self.val_dataloader,
                                            num_samples=self.num_val_samples)
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    self._save_checkpoint()

                    if self.verbose and self.epoch % self.print_acc_every == 0:
                        print(
                            "(Epoch %d / %d) train acc: %f; val_acc: %f"
                            % (self.epoch, self.num_epochs, train_acc, val_acc)
                        )

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



def run_solver(device):

    _, _, class_names = get_dataloaders(batch_size=1)

    model_conv = models.resnet50(weights='IMAGENET1K_V2')
    model_conv.to(device)
    # freeze inner layers:
    for param in model_conv.parameters():
       param.requires_grad = False

    # swap out final fc layer:
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    criterion = nn.CrossEntropyLoss()

    # all params:
    # optimizer_ft = optim.Adam(model_conv.parameters(), lr=0.001)
    # just fc layer
    optimizer_ft = optim.Adam(model_conv.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    solver = Solver(model_conv,
                    batch_size=BATCH_SIZE,
                    criterion=criterion,
                    optimizer=optimizer_ft,
                    print_every=PRINT_EVERY,
                    num_epochs=NUM_EPOCHS,
                    max_data=MAX_DATA,
                    lr_decay=LR_DECAY,
                    num_train_samples=ACC_SAMPLES,
                    num_val_samples=ACC_VAL_SAMPLES,
                    device=device,
                    )

    solver.train(return_best_params=False)

    plt.plot(solver.loss_history, 'o')
    window_width = 20
    cumsum_vec = np.cumsum(np.insert(solver.loss_history, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(list(range(0, len(solver.loss_history) - window_width + 1)), ma_vec)
    plt.show()


    plt.plot(solver.train_acc_history)
    plt.plot(solver.val_acc_history)
    plt.show()
    solver.plot_solver_results(100)
    torch.save(solver, "/home/stoyelq/Documents/dfobot_data/model_solver.pt")