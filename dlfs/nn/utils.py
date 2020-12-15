from numpy import ndarray
import numpy as np
from scipy.special import logsumexp


def assert_same_shape(array_1: ndarray, array_2: ndarray):
    """Ensure proper shapes."""
    msg = f"""
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {tuple(array_1.shape)}
        and second ndarray's shape is {tuple(array_2.shape)}.
        """
    assert array_1.shape == array_2.shape, msg
    return None


def normalize(a: np.ndarray):
    """Normalize a single-class array."""
    other = 1 - a
    return np.concatenate([a, other], axis=1)


def unnormalize(a: np.ndarray):
    """Un-Normalize probabilities for a single-class array."""
    if a.ndim < 1:
        a = a.reshape(-1, 1)
    return a[:, 0, np.newaxis]


def softmax(x, axis=None):
    """Compute softmax transform."""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def calc_accuracy_model(model, test_set, actual):
    acc = (
        np.equal(
            np.argmax(model.forward(test_set, inference=True), axis=1), actual
        ).sum()
        * 100.0
        / test_set.shape[0]
    )
    print(f"The model validation accuracy is: {acc:.2f}")
