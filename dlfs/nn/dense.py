from numpy import ndarray
import numpy as np

from dlfs.nn.core import ParamOperation


class WeightMultiply(ParamOperation):
    """Weight multiplication Operation for a neural network."""

    def __init__(self, weights: ndarray):
        """Initialize Operation with self.param = W."""
        super().__init__(weights)

    def _output(self) -> ndarray:
        """Compute output."""
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input gradient."""
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """Compute parameter gradient."""
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """Compute bias addition."""

    def __init__(self, bias: ndarray):
        """
        Initialize Operation with self.param = B.
        Check appropriate shape.
        """
        assert bias.shape[0] == 1
        super().__init__(bias)

    def _output(self) -> ndarray:
        """Compute output."""
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input gradient."""
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """Compute parameter gradient."""
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
