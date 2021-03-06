from typing import List

from numpy import ndarray

from dlfs.nn.layers import Layer
from dlfs.nn.losses import Loss


class NeuralNetwork:
    """The class for a neural network."""

    def __init__(self, layers: List[Layer], loss: Loss, seed: int = 1):
        """Constructor method."""
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray, inference: bool) -> ndarray:
        """Passes data forward through a series of layers."""
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out, inference)
        return x_out

    def backward(self, loss_grad: ndarray) -> ndarray:
        """Passes data backward through a series of layers."""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        """Gets the parameters for the network."""
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """Gets the gradient of the loss with respect to the parameters for the network."""
        for layer in self.layers:
            yield from layer.param_grads

    def forward_loss(
        self, x_batch: ndarray, y_batch: ndarray, inference: bool = False
    ) -> float:
        """Compute forward loss."""
        # todo this doesnt appear to be used
        prediction = self.forward(x_batch, inference)
        return self.loss.forward(prediction, y_batch)

    def train_batch(
        self, x_batch: ndarray, y_batch: ndarray, inference: bool = False
    ) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        """
        predictions = self.forward(x_batch, inference)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"
