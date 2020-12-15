import numpy as np
from numpy import ndarray

from dlfs.nn.core import Operation


class Dropout(Operation):
    """Dropout operation."""
    def __init__(self, keep_prob: float = 0.8):
        """Constructor method."""
        super().__init__()
        self.keep_prob = keep_prob
        self.mask = None

    def _output(self, inference: bool) -> ndarray:
        """Compute output."""
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input gradient."""
        return output_grad * self.mask
