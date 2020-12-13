from typing import Dict, Tuple, Optional, Union, List

import numpy as np

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

    def _foward_pass(
        self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> "LinearRegressor":
        """Compute forward pass for linear regression."""
        assert x_batch.shape[0] == y_batch.shape[0]
        assert x_batch.shape[1] == self._weights["W"].shape[0]
        assert self._weights["B"].shape[0] == self._weights["B"].shape[1] == 1

        N = np.dot(x_batch, self._weights["W"])
        P = N + self._weights["B"]
        self.loss = np.mean(np.power(y_batch - P, 2))

        self._forward_info["X"] = x_batch
        self._forward_info["N"] = N
        self._forward_info["P"] = P
        self._forward_info["y"] = y_batch
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
