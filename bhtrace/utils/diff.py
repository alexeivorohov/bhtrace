"""
This module provides numerical differentiation procedures using a functional approach.
It is designed to compute derivatives of tensor-valued functions via finite
differences.
"""

from typing import Callable, List, Dict, Any, Protocol
import torch

from bhtrace.utils.registry import CallableRegistry

class DiffScheme(Protocol):
    def __call__(    
        func: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        dx: torch.Tensor,
        eps: float,
        args: List = None,
        kwargs: Dict = None,
    ) -> torch.Tensor:
        pass

DIFF_SCHEME_REGISTRY = CallableRegistry(DiffScheme)

def jacobian(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    order: int = 2,
    args: List = None,
    kwargs: Dict = None,
) -> torch.Tensor:
    """
    Computes the Jacobian of a function func: R^n -> R^m using finite differences.
    The Jacobian tensor will have shape (*x.shape[:-1], *func(x).shape[x.ndim-1:], x.shape[-1]).
    For a function g: R^n -> R^(p x q), and input x of shape (..., n),
    the output Jacobian will have shape (..., p, q, n).

    Parameters
    ----------
    func : callable (..., n) -> (..., any)
        Scalar, vector or tensor function to differentiate
    x : torch.Tensor (..., n)
        The point at which to differentiate
    eps : float (default=1e-4)
        The finite difference step size.
    order : int (default=2),
        The order of the numerical differentiation scheme (1, 2, or 4).
    args:
        Additional positional args for the `func`
    kwargs:
        Additional keyword arguments for the `func`

    Returns
    -------
    torch.Tensor (...., any, n) - the Jacobian

    """
    scheme = DIFF_SCHEME_REGISTRY[order]

    args = args or []
    kwargs = kwargs or {}

    batch_dims = x.ndim - 1
    coord_dim = x.shape[-1]
    dx = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype) * eps

    jac = [scheme(func, x, dx[i], eps, args, kwargs) for i in range(coord_dim)]

    return torch.stack(jac, -1)


# --- Backends ---

@DIFF_SCHEME_REGISTRY.register(1)
def _jacobian_forward_1st(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    dx: torch.Tensor,
    eps: float,
    args: List = None,
    kwargs: Dict = None,
) -> torch.Tensor:

    return (func(x + dx, *args, **kwargs) - func(x, *args, **kwargs)) / eps


@DIFF_SCHEME_REGISTRY.register(2)
def _jacobian_central_2nd(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    dx: torch.Tensor,
    eps: float,
    args: List = None,
    kwargs: Dict = None,
) -> torch.Tensor:

    return (func(x + dx, *args, **kwargs) - func(x - dx, *args, **kwargs)) / (2 * eps)

@DIFF_SCHEME_REGISTRY.register(4)
def _jacobian_central_4th(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    dx: torch.Tensor,
    eps: float,
    args: List = None,
    kwargs: Dict = None,
) -> torch.Tensor:

    return (
        -func(x + 2 * dx, *args, **kwargs)
        + 8 * func(x - dx, *args, **kwargs)
        + -8 * func(x - dx, *args, **kwargs)
        + func(x - 2 * dx, *args, **kwargs)
    ) / (12 * eps)


if __name__ == "__main__":
    func = lambda x: torch.exp(x).sum(dim=-1)
    d_func = lambda x: torch.exp(x)

    x = torch.randn(1, 4)
    dx = jacobian(func, x)
    print(dx.shape)
    print(d_func(x))
    print(dx)
