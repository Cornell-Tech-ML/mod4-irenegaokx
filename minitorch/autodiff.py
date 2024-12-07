from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_forward = list(vals)
    vals_backward = list(vals)

    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon

    f_forward = f(*vals_forward)
    f_backward = f(*vals_backward)

    derivative = (f_forward - f_backward) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the current variable.

        This method is called during backpropagation to add the computed derivative
        to the current variable.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable.

        Returns
        -------
        int:
            The unique identifier of the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Determines whether the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Determines whether the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable.

        Returns
        -------
        Iterable[Variable]:
            An iterable containing the parent variables of the current variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute local derivatives with respect to the parent variables.


        Args:
        ----
        d_output: Any
            The derivative of the output with respect to the current variable.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]:
            An iterable of (parent_variable, local_derivative) tuples for each parent.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    topo_order = []

    def dfs(v: Variable) -> None:
        """Depth-first search to visit all variables."""
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        topo_order.append(v)

    dfs(variable)
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable (Variable): The right-most variable in the computation graph, where backpropagation starts.
        deriv (Any): The initial derivative with respect to the output, which is propagated backward through the computation graph.

    Returns:
    -------
        None

    """
    # TODO: Implement for Task 1.4.
    topo_order = topological_sort(variable)

    derivatives = {variable.unique_id: deriv}

    for var in topo_order:
        d_output = derivatives.get(var.unique_id, 0)

        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_parent in var.chain_rule(d_output):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += d_parent
                else:
                    derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieves the saved tensors stored during the forward pass.

        Returns
        -------
        Tuple[Any, ...]:
            A tuple of saved values from the forward pass.

        """
        return self.saved_values
