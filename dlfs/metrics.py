import numpy as np


def mae(actuals: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Compute mean absolute error."""
    return np.mean(np.abs(preds - actuals))


def mse(actuals: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Compute mean squared error."""
    return np.mean(np.power(preds - actuals, 2))


def rmse(actuals: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Compute root mean squared error."""
    return np.sqrt(mse(actuals, preds))
