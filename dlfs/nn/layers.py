from typing import List, Union

import numpy as np
from numpy import ndarray

from dlfs.nn.core import Operation, ParamOperation
from dlfs.nn.activations import Sigmoid
from dlfs.nn.dense import WeightMultiply, BiasAdd
from dlfs.nn.utils import assert_same_shape


class Layer:
    """A "layer" of neurons in a neural network."""

    def __init__(self, neurons: int):
        """
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        """
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []
        self.input_ = None
        self.output = None

    def _setup_layer(self, num_in: Union[int, ndarray]) -> None:
        """The _setup_layer function must be implemented for each layer."""
        raise NotImplementedError()

    def forward(self, input_: ndarray, inference: bool = False) -> ndarray:
        """Passes input forward through a series of operations."""
        if self.first:
            self._setup_layer(input_)
            self.first = False
        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_, inference)
        self.output = input_
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """Passes output_grad backward through a series of operations."""
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self):
        """Extracts the _param_grads from a layer's operations."""
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        """Extracts the _params from a layer's operations."""
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """A fully connected layer which inherits from Layer."""

    def __init__(self, neurons: int, activation: Operation = Sigmoid()):
        """Constructor method."""
        super().__init__(neurons)
        self.activation = activation
        self.seed = None

    def _setup_layer(self, input_: ndarray) -> None:
        """Defines the operations of a fully connected layer."""
        if self.seed:
            np.random.seed(self.seed)

        initial_weights = np.random.randn(input_.shape[1], self.neurons)
        initial_bias = np.random.randn(1, self.neurons)
        self.params = [initial_weights, initial_bias]
        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]
        return None
