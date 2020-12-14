import numpy as np


def mae(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(preds - actuals)))


def mse(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean(np.power(preds - actuals, 2)))


def rmse(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mse(actuals, preds)))
