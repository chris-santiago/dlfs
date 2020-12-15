import numpy as np


class Optimizer:
    """Base class for a neural network optimizer."""

    def __init__(self, lr: float = 0.01, final_lr: float = 0.0, decay_type: str = None):
        """Every optimizer must have an initial learning rate."""
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True
        self.max_epochs = 50
        self.net = None
        self.decay_per_epoch = None

    def setup_decay(self) -> None:
        """Setup decay for optimizer."""
        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr / self.lr, 1.0 / (self.max_epochs - 1)
            )
        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def decay_lr(self) -> None:
        """Decay the learning rate."""
        if not self.decay_type:
            return
        if self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch
        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

    def step(self) -> None:
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the
        adjustment based on the learning rate.
        """
        if self.net:
            for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
                self._update_rule(param=param, grad=param_grad)
        raise AttributeError("Net attribute cannot be empty.")

    def _update_rule(self, **kwargs) -> None:
        """Each Optimizer must implement an update rule."""
        raise NotImplementedError()


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, lr: float = 0.01) -> None:
        """Constructor method."""
        super().__init__(lr)

    def _update_rule(self, **kwargs) -> None:
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the
        adjustment based on the learning rate.
        """
        kwargs["param"] -= self.lr * kwargs["grad"]


class SGDMomentum(Optimizer):
    """Stochastic gradient descent optimizer with momentum."""

    def __init__(
        self,
        lr: float = 0.01,
        final_lr: float = 0,
        decay_type: str = None,
        momentum: float = 0.9,
    ):
        """Constructor method."""
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum
        self.velocities = None

    def step(self) -> None:
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the
        adjustment based on the learning rate.
        """
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(
            self.net.params(), self.net.param_grads(), self.velocities
        ):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:
        """Update velocity and parameters."""
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]
        kwargs["param"] -= kwargs["velocity"]
