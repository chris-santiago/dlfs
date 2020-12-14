from numpy import ndarray
import numpy as np

from dlfs.nn.core import Operation


class Sigmoid(Operation):
    """Sigmoid activation function."""

    def __init__(self):
        """Constructor method."""
        super().__init__()

    def _output(self) -> ndarray:
        """Compute output."""
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input gradient."""
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """Identity activation function."""

    def __init__(self):
        """Constructor method."""
        super().__init__()

    def _output(self) -> ndarray:
        """Pass through."""
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Pass through."""
        return output_grad
