from typing import Tuple
from copy import deepcopy

import numpy as np
from numpy import ndarray

from dlfs.nn.network import NeuralNetwork
from dlfs.nn.optimizers import Optimizer


class Trainer:
    """Trains a neural network."""

    def __init__(self, net: NeuralNetwork, optim: Optimizer, batch_size: int = 32):
        """
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.batch_size = batch_size
        self.best_loss = 1e9
        setattr(self.optim, "net", self.net)

    @staticmethod
    def permute_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly permute X and y."""
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    @staticmethod
    def generate_batches(
        X: ndarray, y: ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates batches for training."""
        assert X.shape[0] == y.shape[0]

        for start in range(0, X.shape[0], batch_size):
            x_batch, y_batch = (
                X[start : start + batch_size],
                y[start : start + batch_size],
            )
            yield x_batch, y_batch

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        x_valid: ndarray,
        y_valid: ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        single_output: bool = False,
        restart: bool = True,
        early_stopping: bool = True,
        conv_testing: bool = False,
    ) -> None:
        """
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        """
        setattr(self.optim, "max_epochs", epochs)
        self.optim.setup_decay()

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        for epoch in range(epochs):
            if (epoch + 1) % eval_every == 0:
                last_model = deepcopy(self.net)  # for early stopping
            x_train, y_train = self.permute_data(x_train, y_train)
            batch_generator = self.generate_batches(x_train, y_train, batch_size)

            for _, (x_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(x_batch, y_batch)
                self.optim.step()

            if (epoch + 1) % eval_every == 0:
                valid_preds = self.net.forward(x_valid, inference=True)
                loss = self.net.loss.forward(valid_preds, y_valid)
                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {epoch + 1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print(
                            f"Loss increased after epoch {epoch + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {epoch + 1 - eval_every}"
                        )
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, "net", self.net)
                        break
                else:
                    print(f"Validation loss after {epoch + 1} epochs is {loss:.3f}")
            if self.optim.final_lr:
                self.optim.decay_lr()
