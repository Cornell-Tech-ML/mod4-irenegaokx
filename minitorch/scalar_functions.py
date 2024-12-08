from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward operation to the provided input values.

        This method processes the given values and converts them into Scalar objects,
        and stores their raw data for use in the forward operation.

        Args:
        ----
            *vals:
                A variable number of input values, which can be either `Scalar` objects or
                other scalar-like types (e.g., `float`, `int`).

        Returns:
        -------
            Scalar:
                A new `Scalar` object that represents the result of applying the forward
                operation on the inputs.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two scalars."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients of the addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two scalars."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients of the multiplication."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of a scalar."""
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse function."""
        (a,) = ctx.saved_values
        return -1.0 / (a**2) * d_output


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negate a scalar."""
        result = -float(a)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid function."""
        result = 1.0 / (1.0 + operators.exp(-a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the sigmoid function."""
        (result,) = ctx.saved_values
        return result * (1 - result) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of a scalar."""
        ctx.save_for_backward(a)
        return max(0.0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the ReLU function."""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of a scalar."""
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the exponential function."""
        (result,) = ctx.saved_values
        return result * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compare if a is less than b."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No gradient needed for less-than comparison."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compare if a is equal to b."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No gradient needed for equality comparison."""
        return 0.0, 0.0
