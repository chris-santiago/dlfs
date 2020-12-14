from typing import List, Union, Tuple
from copy import deepcopy

import numpy as np
from numpy import ndarray

from dlfs.metrics import mae, rmse


def assert_same_shape(array_1: np.ndarray, array_2: np.ndarray):
    """Ensure proper shapes."""
    msg = f"""
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {tuple(array_1.shape)}
        and second ndarray's shape is {tuple(array_2.shape)}.
        """
    assert array_1.shape == array_2.shape, msg
    return None


class Operation:
    """Base class for an Operation in a neural network."""

    def __init__(self):
        self.input_ = None
        self.output = None
        self.input_grad = None

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

    def forward(self, input_: ndarray) -> ndarray:
        """Passes input forward through a series of operations."""
        if self.first:
            self._setup_layer(input_)
            self.first = False
        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
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

    def forward(self, x_batch: ndarray) -> ndarray:
        """Passes data forward through a series of layers."""
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        """Passes data backward through a series of layers."""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        """
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        """Gets the parameters for the network."""
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """Gets the gradient of the loss with respect to the parameters for the network."""
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer:
    """Base class for a neural network optimizer."""

    def __init__(self, lr: float = 0.01):
        """Every optimizer must have an initial learning rate."""
        self.lr = lr

    def step(self) -> None:
        """Every optimizer must implement the "step" function."""
        pass


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, lr: float = 0.01) -> None:
        """Constructor method."""
        super().__init__(lr)
        self.net = None

    def step(self):
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the
        adjustment based on the learning rate.
        """
        if self.net:
            for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
                param -= self.lr * param_grad
        else:
            raise AttributeError("Net attribute cannot be empty.")


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
        restart: bool = True,
    ) -> None:
        """
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        """
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

            for batch_num, (x_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(x_batch, y_batch)
                self.optim.step()

            if (epoch + 1) % eval_every == 0:
                valid_preds = self.net.forward(x_valid)
                loss = self.net.loss.forward(valid_preds, y_valid)
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


def eval_regression_model(model: NeuralNetwork, x_test: ndarray, y_test: ndarray):
    """Evaluate neural network regressor using MAE and RMSE."""
    preds = model.forward(x_test).reshape(-1, 1)
    print(
        f"""
        Mean absolute error: {round(mae(y_test, preds), 2)} \n
        Root mean squared error: {round(rmse(y_test, preds), 2)}
        """
    )
