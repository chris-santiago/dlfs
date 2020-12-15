import numpy as np


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


class SGDMomentum(Optimizer):
    """Stochastic gradient descent optimizer with momentum."""
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9):
        """Constructor method."""
        super().__init__(lr)
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.momentum = momentum
        self.velocities = None
        self.first = None
        self.net = None

    def step(self) -> None:
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the
        adjustment based on the learning rate.
        """
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:
        """Update velocity and parameters."""
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['grad']
        kwargs['param'] -= kwargs['velocity']
