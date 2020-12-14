class Optimizer(object):
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
