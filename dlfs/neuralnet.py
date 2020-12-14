from typing import List

import numpy as np
from numpy import ndarray


def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    """Ensure proper shapes."""
    msg = f'''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {tuple(array_grad.shape)}
        and second ndarray's shape is {tuple(array.shape)}.
        '''
    assert array.shape == array_grad.shape, msg
    return None


class Operation:
    """Base class for an Operation in a neural network."""
    def __init__(self):
        self.input_ = None
        self.output = None
        self.input_grad= None

    def forward(self, input_: ndarray):
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        """The _output method must be defined for each Operation."""
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """The _input_grad method must be defined for each Operation."""
        raise NotImplementedError()


class ParamOperation(Operation):
    """An Operation with parameters."""
    def __init__(self, param: ndarray):
        """Constructor method."""
        super().__init__()
        self.param = param
        self.param_grad = None

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """Every subclass of ParamOperation must implement _param_grad."""
        raise NotImplementedError()


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
