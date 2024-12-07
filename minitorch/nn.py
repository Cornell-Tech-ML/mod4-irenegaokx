from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand
#correct

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    #raise NotImplementedError("Need to implement for Task 4.3")
    new_height = height // kh
    new_width = width // kw

    # Reshape and permute to create the tiled structure
    input = input.contiguous()
    tiled = input.view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        A tensor of size batch x channel x new_height x new_width

    """
    # Reshape the input using the tile function
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average along the kernel dimension (last dimension)
    pooled = tiled.mean(dim=-1)

    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)



max_reduce = FastOps.reduce(operators.max, -1e9)



def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension.

    Args:
    ----
        input: The input tensor.
        dim: The dimension along which to compute the argmax.

    Returns:
    -------
        A one-hot encoded tensor with the same shape as the input, where the
        max index along the specified dimension is set to 1 and others to 0.

    """
    max_values = max_reduce(input, dim)

    return max_values == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max"""
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max"""
        input, dim = ctx.saved_values
        arg_max = argmax(input, dim)
        return (grad_output * arg_max, 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


# TODO: Implement for Task 4.3.

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as tensor.

    Args:
    ----
        input: batch x channel x height x width
        dim: The dimension to compute softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with softmax

    """
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax as a tensor.

    Args:
    ----
        input: batch x channel x height x width
        dim: The dimension to compute log softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with log softmax applied

    """
    # Max value for numerical stability
    max_input = max(input, dim)
    
    # Subtract the max and exponentiate, sum, and take the log
    stable_input = input - max_input
    log_sum_exp = (stable_input.exp().sum(dim)).log()
    
    # Compute log softmax
    return stable_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max pooling on a 2D input tensor using a specified kernel size.

    This function first tiles the input tensor according to the kernel size,
    then reduces each tile to its maximum value, and finally reshapes the output
    to the appropriate dimensions.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width).
        kernel: Tuple specifying the height and width of the pooling window.

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width) after max pooling.
        
    """
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    # Apply a max reduction across the last dimension of the tiled tensor
    # which corresponds to the flattened pooling windows
    reduced = max_reduce(tiled, -1).contiguous()
    # Reshape the reduced tensor to the new dimensions, matching the output
    # of a max pooling operation
    return reduced.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor by randomly setting a fraction of the elements to zero.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        p (float): Probability of an element to be zeroed.
        ignore (bool): Flag to ignore dropout and return the input unchanged. Useful during evaluation.

    Returns:
    -------
        Tensor: Output tensor after applying dropout. Elements are zeroed with probability `p`.

    """
    if ignore:
        return input
    if p == 1:
        return input.zeros(input.shape)
    if p == 0:
        return input
    else:
        mask = (rand(input.shape) > p) # Generate a random tensor and compare each element to p
        
        # Scale the active elements to maintain overall activation scale
        scale = 1.0 / (1.0 - p)

        # Apply the mask to the input tensor and scale appropriately
        return input * mask * scale