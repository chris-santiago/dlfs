from numpy import ndarray

from dlfs.nn.utils import assert_same_shape


class Operation:
    """Base class for an Operation in a neural network."""

    def __init__(self):
        self.input_ = None
        self.output = None
        self.input_grad = None

    def forward(self, input_: ndarray, inference: bool = False):
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        """
        self.input_ = input_
        self.output = self._output(inference)
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        """The _output method must be defined for each Operation."""
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """The _input_grad method must be defined for each Operation."""
        raise NotImplementedError()


class ParamOperation(Operation):
    """An Operation with parameters."""

    def __init__(self, param: ndarray):
        """Constructor method."""
        super().__init__()
        self.param = param
        self.param_grad = None

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """Every subclass of ParamOperation must implement _param_grad."""
        raise NotImplementedError()
