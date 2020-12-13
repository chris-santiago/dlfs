from typing import Dict, Tuple, Optional, Union, List

import numpy as np

from dlfs.base import sigmoid
from dlfs.metrics import mse

Batch = Tuple[np.ndarray, np.ndarray]


class LinearRegressor:
    """Class for manual linear regression."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = 100,
        n_iter: int = 1000,
        return_losses: bool = False,
        return_weights: bool = False,
        seed: int = 1,
    ):
        """Initialize object."""
        self._eta = learning_rate
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._return_losses = return_losses
        self._return_weights = return_weights
        self._seed = seed
        self._forward_info: Dict[str, np.ndarray] = {}
        self._loss_gradients: Dict[str, np.ndarray] = {}
        self._weights: Dict[str, np.ndarray] = {}
        self.loss: Optional[float] = None

    @staticmethod
    def _permute_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly permute X and y."""
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    def _make_batch(self, X: np.ndarray, y: np.ndarray, start: int = 0) -> Batch:
        """Generate batch from X and y, given start position."""
        assert X.ndim == y.ndim == 2

        if not self._batch_size:
            self._batch_size = X.shape[0]

        x_batch, y_batch = (
            X[start : start + self._batch_size],
            y[start : start + self._batch_size],
        )
        return x_batch, y_batch

    def _initialize_weights(self, n_weights: int) -> "LinearRegressor":
        """Initialize weights for first forward pass."""
        self._weights["W"] = np.random.randn(n_weights, 1)
        self._weights["B"] = np.random.randn(1, 1)
        return self

    def _get_forward_loss(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Generate fitted values and compute loss. """
        fitted_no_bias = np.dot(x_batch, self._weights["W"])
        fitted_with_bias = fitted_no_bias + self._weights["B"]
        self.loss = float(np.mean(np.power(y_batch - fitted_with_bias, 2)))
        self._forward_info["X"] = x_batch
        self._forward_info["N"] = fitted_no_bias
        self._forward_info["P"] = fitted_with_bias
        self._forward_info["y"] = y_batch
        return self.loss

    def _get_loss_gradients(self) -> "LinearRegressor":
        """Compute gradients for linear regression."""
        dL_dP = -2 * (self._forward_info["y"] - self._forward_info["P"])
        dP_dN = np.ones_like(self._forward_info["N"])
        dP_dB = np.ones_like(self._weights["B"])
        dL_dN = dL_dP * dP_dN
        dN_dW = np.transpose(self._forward_info["X"], (1, 0))
        dL_dW = np.dot(dN_dW, dL_dN)
        dL_dB = (dL_dP * dP_dB).sum(axis=0)

        self._loss_gradients["W"] = dL_dW
        self._loss_gradients["B"] = dL_dB
        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> Union[
        Tuple[List[float], Dict[str, np.ndarray]], List[float], "LinearRegressor"
    ]:
        """Fit regression model."""
        if self._seed:
            np.random.seed(self._seed)
        start = 0
        self._initialize_weights(X.shape[1])
        X, y = self._permute_data(X, y)
        losses = []
        for i in range(self._n_iter):
            if start >= X.shape[0]:
                X, y = self._permute_data(X, y)
                start = 0
            x_batch, y_batch = self._make_batch(X, y, start)
            start += self._batch_size
            self.loss = self._get_forward_loss(x_batch, y_batch)
            if self._return_losses:
                losses.append(self.loss)
            self._get_loss_gradients()
            for param in self._weights:
                self._weights[param] -= self._eta * self._loss_gradients[param]
        if self._return_weights:
            return losses, self._weights
        if self._return_losses:
            return losses
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from regression model."""
        return np.dot(X, self._weights["W"]) + self._weights["B"]


class NNRegressor(LinearRegressor):
    """Class for manual neural network regressor."""

    def __init__(
            self,
            hidden_size: int,
            learning_rate: float = 0.01,
            batch_size: Optional[int] = 100,
            n_iter: int = 1000,
            validate_step: int = 1000,
            return_losses: bool = False,
            return_weights: bool = False,
            return_scores: bool = False,
            seed: int = 1,
    ):
        self.hidden_size = hidden_size
        self.validate_step = validate_step
        self._return_scores = return_scores
        super().__init__(learning_rate, batch_size, n_iter, return_losses, return_weights, seed)

    def _initialize_weights(self, input_size: int) -> "NNRegressor":
        """Initialize weights for first forward pass."""
        self._weights["W1"] = np.random.randn(input_size, self.hidden_size)
        self._weights["B1"] = np.random.randn(1, self.hidden_size)
        self._weights["W2"] = np.random.randn(self.hidden_size, 1)
        self._weights["B2"] = np.random.randn(1, 1)
        return self

    def _get_forward_loss(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Generate fitted values and compute loss. """
        input_fit_no_bias = np.dot(x_batch, self._weights["W1"])
        input_fit_with_bias = input_fit_no_bias + self._weights["B1"]
        activation_fit = sigmoid(input_fit_with_bias)
        hidden_fit_no_bias = np.dot(activation_fit, self._weights['W2'])
        hidden_fit_with_bias = hidden_fit_no_bias + self._weights['B2']
        self.loss = float(np.mean(np.power(y_batch - hidden_fit_with_bias, 2)))
        self._forward_info["X"] = x_batch
        self._forward_info["M1"] = input_fit_no_bias
        self._forward_info["N1"] = input_fit_with_bias
        self._forward_info["O1"] = activation_fit
        self._forward_info["M2"] = hidden_fit_no_bias
        self._forward_info["P"] = hidden_fit_with_bias
        self._forward_info["y"] = y_batch
        return self.loss

    def _get_loss_gradients(self) -> "NNRegressor":
        """Compute gradients for NN regression."""
        dL_dP = -2 * (self._forward_info["y"] - self._forward_info["P"])
        dP_dM2 = np.ones_like(self._forward_info['M2'])
        dL_dM2 = dL_dP * dP_dM2
        dP_dB2 = np.ones_like(self._weights['B2'])
        dL_dB2 = (dL_dP * dP_dB2).sum(axis=0)

        dM2_dW2 = np.transpose(self._forward_info['O1'], (1, 0))
        dL_dW2 = np.dot(dM2_dW2, dL_dP)

        dM2_dO1 = np.transpose(self._weights['W2'], (1, 0))
        dL_dO1 = np.dot(dL_dM2, dM2_dO1)
        dO1_dN1 = sigmoid(self._forward_info['N1']) * (1 - sigmoid(self._forward_info['N1']))
        dL_dN1 = dL_dO1 * dO1_dN1
        dN1_dB1 = np.ones_like(self._weights['B1'])
        dN1_dM1 = np.ones_like(self._forward_info['M1'])
        dL_dB1 = (dL_dN1 * dN1_dB1).sum(axis=0)

        dL_dM1 = dL_dN1 * dN1_dM1
        dM1_dW1 = np.transpose(self._forward_info['X'], (1, 0))
        dL_dW1 = np.dot(dM1_dW1, dL_dM1)

        self._loss_gradients["W2"] = dL_dW2
        self._loss_gradients["B2"] = dL_dB2.sum(axis=0)
        self._loss_gradients["W1"] = dL_dW1
        self._loss_gradients["B1"] = dL_dB1.sum(axis=0)
        return self

    def fit(
            self, X: np.ndarray, y: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray
    ) -> Union[
        Tuple[List[float], Dict[str, np.ndarray], List[float]], List[float], Tuple[List[float], List[float]], "LinearRegressor"
    ]:
        """Fit regression model."""
        if self._seed:
            np.random.seed(self._seed)
        start = 0
        self._initialize_weights(X.shape[1])
        X, y = self._permute_data(X, y)
        losses = []
        val_scores = []
        for i in range(self._n_iter):
            if start >= X.shape[0]:
                X, y = self._permute_data(X, y)
                start = 0
            x_batch, y_batch = self._make_batch(X, y, start)
            start += self._batch_size
            self.loss = self._get_forward_loss(x_batch, y_batch)
            if self._return_losses:
                losses.append(self.loss)
            self._get_loss_gradients()
            for param in self._weights:
                self._weights[param] -= self._eta * self._loss_gradients[param]
            if self._return_scores:
                if (i % self.validate_step == 0) and (i != 0):
                    preds = self.predict(x_valid)
                    val_scores.append(float(mse(y_valid, preds)))
        if self._return_weights:
            return losses, self._weights, val_scores
        if self._return_scores:
            return losses, val_scores
        if self._return_losses:
            return losses
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the step-by-step neural network model."""
        M1 = np.dot(X, self._weights['W1'])
        N1 = M1 + self._weights['B1']
        O1 = sigmoid(N1)
        M2 = np.dot(O1, self._weights['W2'])
        return M2 + self._weights['B2']
