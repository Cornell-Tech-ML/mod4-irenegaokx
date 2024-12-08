from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other submodules.

    They make up the basis of neural network stacks.

    Attributes
    ----------
    _modules : Dict[str, Module]
        Storage of the child modules.
    _parameters : Dict[str, Parameter]
        Storage of the module's parameters.
    training : bool
        Whether the module is in training mode or evaluation mode.

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to `train`."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to `eval`."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
        Sequence[Tuple[str, Parameter]]
            The name and `Parameter` of each ancestor parameter.

        """
        params = []
        for name, param in self._parameters.items():
            params.append((name, param))
        for module_name, module in self._modules.items():
            params.extend(
                [
                    (module_name + "." + name, param)
                    for name, param in module.named_parameters()
                ]
            )
        return params

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
        k : str
            Local name of the parameter.
        v : Any
            Value for the parameter.

        Returns:
        -------
        Parameter
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the module's `forward` method.

        Args:
        ----
        *args : Any
            Positional arguments to pass to the `forward` method.
        **kwargs : Any
            Keyword arguments to pass to the `forward` method.

        Returns:
        -------
        Any
            The output of the module's `forward` method.

        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the module.

        This representation includes all submodules and their structures.

        Returns
        -------
        str
            A string representing the module and its submodules.

        """

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.

    Attributes
    ----------
    value : Any
        The value of the parameter, which can be a `Variable` or any type.
    name : Optional[str]
        An optional name for the parameter.

    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value.

        Args:
        ----
        x : Any
            The new value to update the parameter to.

        """
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """Return a string representation of the parameter's value."""
        return repr(self.value)

    def __str__(self) -> str:
        """Return a string representation of the parameter's value."""
        return str(self.value)
