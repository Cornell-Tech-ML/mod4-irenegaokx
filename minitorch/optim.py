from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Resets the gradients of all parameters in the model to zero.

        Iterates through all the parameters of the model and sets their `derivative`
        or `grad` attributes to None, clearing out the accumulated gradients.

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Updates the model's parameters based on the computed gradients.

        For each parameter, it subtracts the product of the learning rate (`self.lr`)
        and the parameter's gradient (either `derivative` or `grad`). The parameter
        is updated in place to reflect the new value after the gradient step.

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
