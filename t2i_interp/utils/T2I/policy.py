from typing import Callable, Optional, Union, Tuple, List, Any, Dict
import torch as t
from t2i_interp.utils.generic import reshape_like

Tensor = t.Tensor

def scale_policy(factor: Union[float, Tensor]) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    factor: float or tensor. If tensor, it will be broadcast to x.
    """
    if not t.is_tensor(factor):
        factor = t.tensor(factor)

    def _f(x: Tensor, module: t.nn.Module) -> Tensor:
        f = reshape_like(factor, x)
        return x * f
    return _f

def add_vector_policy(vec: Tensor, *, channel_dim: Optional[int] = None) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    vec: tensor to add; can be 1D (e.g., channels) or any shape broadcastable to x.
    channel_dim: set if you want to force which dimension the 1D vec maps to.
    """
    def _f(x: Tensor, module: t.nn.Module) -> Tensor:
        v = reshape_like(vec, x)
        return x + v
    return _f

def replace_policy(
    # value: Union[Tensor, Callable[[Tensor, t.nn.Module], Tensor]]
) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    value: 
      - Tensor: used directly (broadcast to x if needed)
      - Callable: called as value(x, module) -> Tensor
    """
    def _f(x: Tensor, **kwargs) -> Tensor:
        value = kwargs.get("value", None)
        if value is None:
            return x
        new = value(x) if callable(value) else value
        new = reshape_like(new, x)
        return new
    return _f

def scale_indx_policy(scale: float, dims: List[int]) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that scales the specified dimensions in the latent vector z.
    Usage: z_alter_fn = scale_indx_policy(0.5, [1, 5, 10])
    """
    def fn(z: Tensor) -> Tensor:
        z = z.clone()
        z[:, dims] *= scale
        return z
    return fn