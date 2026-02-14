"""
This module provides numerical differentiation procedures using a functional approach.
It is designed to compute derivatives of tensor-valued functions via finite
differences.
"""

from typing import Callable, List, Dict
import torch


def jacobian(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    order: int = 2,
    args: List = None,
    kwargs: Dict = None
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
    args = args or []
    kwargs = kwargs or {}

    match order:
        case 1: return _jacobian_forward_1st(func, x, eps, args, kwargs)
        case 2: return _jacobian_central_2nd(func, x, eps, args, kwargs)
        case 4: return _jacobian_central_4th(func, x, eps, args, kwargs)
        case _:
            raise ValueError(
                f"Invalid order: {order}. Supported orders are 1, 2, and 4."
            )

# --- Backends ---

def _fetch_jacobian(jac: torch.Tensor, n_batch_dims: int) -> torch.Tensor:
    ...
    if n_batch_dims == jac.ndim - 1:
        return jac

    perm = list(range(jac.ndim))
    perm.append(perm.pop(n_batch_dims))

    return jac.permute(perm)

def _jacobian_forward_1st(    
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    args: List = None,
    kwargs: Dict = None
) -> torch.Tensor:
    
    batch_dims = x.ndim - 1
    coord_dim = x.shape[-1]
    dx = torch.eye(
        x.shape[-1], device=x.device, dtype=x.dtype
        ).view(*[1 for _ in range(batch_dims)], coord_dim, coord_dim)*eps
    x = x.unsqueeze(-1)

    jac = (func(x + dx, *args, **kwargs) - func(x, *args, **kwargs))/eps

    return _fetch_jacobian(jac, batch_dims)


def _jacobian_central_2nd(    
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    args: List = None,
    kwargs: Dict = None
) -> torch.Tensor:
    
    batch_dims = x.ndim - 1
    coord_dim = x.shape[-1]
    dx = torch.eye(
        x.shape[-1], device=x.device, dtype=x.dtype
        ).view(*[1 for _ in range(batch_dims)], coord_dim, coord_dim)*eps
    x = x.unsqueeze(-1)

    jac = (func(x + dx, *args, **kwargs) - func(x - dx, *args, **kwargs))/(2*eps)

    return _fetch_jacobian(jac, batch_dims)


def _jacobian_central_4th(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    args: List = None,
    kwargs: Dict = None
) -> torch.Tensor:
    
    batch_dims = x.ndim - 1
    coord_dim = x.shape[-1]
    dx = torch.eye(
        x.shape[-1], device=x.device, dtype=x.dtype
        ).view(*[1 for _ in range(batch_dims)], coord_dim, coord_dim)*eps
    x = x.unsqueeze(-1)

    jac = (
        - func(x + 2*dx, *args, **kwargs) + 8*func(x - dx, *args, **kwargs) + \
        - 8*func(x - dx, *args, **kwargs) + func(x - 2*dx, *args, **kwargs)
        )/(12*eps)

    return _fetch_jacobian(jac, batch_dims)

if __name__ == '__main__':
    func = lambda x: torch.exp(x).sum(dim=-1)
    d_func = lambda x: torch.exp(x)

    x = torch.randn(1,4)
    dx = jacobian(func, x)
    print(dx.shape)
    print(d_func(x))
    print(dx)