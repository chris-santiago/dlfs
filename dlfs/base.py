from typing import Callable, List, Tuple

import numpy as np


ArrayFunction = Callable[[np.ndarray], np.ndarray]

Chain = List[ArrayFunction]


def square(x: np.ndarray) -> np.ndarray:
    """Square each element in an array."""
    return np.power(x, 2)


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Apply leaky RELU function to each element in an array."""
    return np.maximum(0.2 * x, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function to each element in an array."""
    return 1 / (1 + np.exp(-x))


def get_derivative(func: ArrayFunction, input_: np.ndarray, delta: float = 0.001) -> np.ndarray:
    """Evaluate the derivative of a function at every element in the input array."""
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def eval_two_chain(chain: Chain, x: np.ndarray) -> np.ndarray:
    """Evaluate two functions in a chain."""
    assert len(chain) == 2
    func_1 = chain[0]
    func_2 = chain[1]
    return func_2(func_1(x))


def eval_three_chain(chain: Chain, x: np.ndarray) -> np.ndarray:
    """Evaluate two functions in a chain."""
    assert len(chain) == 3
    func_1 = chain[0]
    func_2 = chain[1]
    func_3 = chain[2]
    return func_3(func_2(func_1(x)))


def deriv_two_chain(chain: Chain, x: np.ndarray) -> np.ndarray:
    """Use chain rule to get derivative of two nested functions."""
    assert len(chain) == 2
    assert x.ndim == 1

    func_1 = chain[0]
    func_2 = chain[1]
    f1_of_x = func_1(x)
    df1_dx = get_derivative(func_1, x)
    df2_du = get_derivative(func_2, f1_of_x)
    return df1_dx * df2_du


def deriv_three_chain(chain: Chain, x: np.ndarray) -> np.ndarray:
    """Use chain rule to get derivative of three nested functions."""
    assert len(chain) == 3
    assert x.ndim == 1

    func_1 = chain[0]
    func_2 = chain[1]
    func_3 = chain[2]
    f1_of_x = func_1(x)
    f2_of_x = func_2(f1_of_x)
    df3_dx = get_derivative(func_3, f2_of_x)
    df2_dx = get_derivative(func_2, f1_of_x)
    df1_df = get_derivative(func_1, x)
    return df1_df * df2_dx * df3_dx


def multiple_inputs_add(x: np.ndarray, y: np.ndarray, sigma: ArrayFunction) -> np.ndarray:
    """Apply function to multiple inputs with addition."""
    return sigma(x+y)


def multiple_inputs_add_backward(x: np.ndarray, y: np.ndarray, sigma: ArrayFunction) -> Tuple[np.ndarray, np.ndarray]:
    """Compute derivative of function with respect to both inputs."""
    a = x+y
    ds_da = get_derivative(sigma, a)
    da_dx, da_dy = 1, 1
    return ds_da * da_dx, ds_da * da_dy


def matmul_forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute forward pass of matrix multiplication."""
    assert x.shape[1] == w.shape[0]
    return np.dot(x, w)


def matmul_backward_first(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute backward pass of matrix multiplication wrt first argument."""
    return np.transpose(w, (1, 0))


