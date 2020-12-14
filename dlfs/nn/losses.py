from numpy import ndarray
import numpy as np

from dlfs.nn.utils import assert_same_shape


class Loss:
    """The loss function for a neural network."""

    def __init__(self):
        """Constructor method."""
        self.prediction = None
        self.target = None
        self.input_grad = None

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """Computes the actual loss value."""
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self) -> ndarray:
        """Computes gradient of the loss value with respect to the input to the loss function."""
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        """Every subclass of "Loss" must implement the _output function."""
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        """Every subclass of "Loss" must implement the _input_grad function."""
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """Mean squared error loss."""

    def __init__(self) -> None:
        """Constructor method."""
        super().__init__()

    def _output(self) -> float:
        """Computes the per-observation squared error loss."""
        loss = (
            np.sum(np.power(self.prediction - self.target, 2))
            / self.prediction.shape[0]
        )
        return loss

    def _input_grad(self) -> ndarray:
        """Computes the loss gradient with respect to the input for MSE loss."""
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
