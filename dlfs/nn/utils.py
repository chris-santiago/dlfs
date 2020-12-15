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


def softmax(x, axis=None):
    """Softmax function."""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


# def normalize(a: np.ndarray):
#     """Normalize an array."""
#     other = 1 - a
#     return np.concatenate([a, other], axis=1)
#
#
# def unnormalize(a: np.ndarray):
#     """Un-Normalize an array."""
#     return a[np.newaxis, 0]
