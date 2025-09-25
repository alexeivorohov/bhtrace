"""
This module provides numerical differentiation procedures used by the library.

It defines an abstract base class `Diff` and concrete implementations for
numerical differentiation schemes of first, second, and fourth order.
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable, Any


class Diff(ABC):
    """
    Abstract base class for numerical differentiation.

    Defines the interface for numerical differentiation methods.
    """

    def __init__(self,
                 func: Callable,
                 eps: float = 1e-6):
        """
        Initialize the differentiation instance.

        Parameters:
        - func: Callable
            The function to differentiate.
        - eps: float, optional
            Step size for numerical differentiation (default: 1e-6).
        """
        self.func = func
        self.eps = eps

    @abstractmethod
    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the numerical derivative of the function at input X.

        Parameters:
        - X: torch.Tensor
            Input values where the derivative is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            The computed derivative values.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class D(Diff):
    """
    Numerical differentiation using finite difference schemes.

    This class acts as a factory that returns an instance of the appropriate
    differentiation scheme based on the `order` parameter.

    Parameters:
    - func: Callable
        The function to be differentiated.
    - eps: float, optional
        Step size for numerical differentiation (default: 1e-5).
    - order: int, optional
        Order of the numerical differentiation scheme.
        Supported values: 1 (first-order), 2 (second-order), 4 (fourth-order).
        Default is 1.
    """

    def __new__(cls,
                func: Callable,
                eps: float = 1e-5,
                order: int = 1) -> Diff:
        if order == 1:
            instance = super().__new__(cls)
        elif order == 2:
            instance = super().__new__(_D_2order_)
        elif order == 4:
            instance = super().__new__(_D_4order_)
        else:
            raise ValueError(f"Invalid order: {order}. Supported orders are 1, 2, and 4.")
        return instance

    def __init__(self,
                 func: Callable,
                 eps: float = 1e-5,
                 order: int = 1):
        # Note: __init__ may be called multiple times due to __new__ returning different classes.
        # To avoid reinitialization, check if attributes already exist.
        if hasattr(self, 'func'):
            return
        super().__init__(func, eps)
        self.order = order

    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the first-order numerical derivative using forward difference.

        Parameters:
        - X: torch.Tensor
            Input values where the derivative is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            The computed derivative values.
        """
        dX = self.eps
        return (self.func(X + dX, *args, **kwargs) - self.func(X, *args, **kwargs)) / dX

    def __check__(self, X: torch.Tensor) -> None:
        """
        Check consistency of input, function output, and derivative shapes.

        Raises:
        - ValueError: If shapes are not aligned as expected.
        """
        y = self.func(X).squeeze()
        dy = self.__call__(X).squeeze()
        X_squeezed = X.squeeze()

        print(f'Input shape: {X_squeezed.shape}')
        print(f'Function output shape: {y.shape}')
        print(f'Derivative output shape: {dy.shape}')

        if X_squeezed.shape != y.shape:
            raise ValueError(f'Input shape {X_squeezed.shape} and function output shape {y.shape} do not align. '
                             f'Maybe the function is not scalar-valued?')

        if X_squeezed.shape != dy.shape:
            raise ValueError(f'Input shape {X_squeezed.shape} and derivative shape {dy.shape} do not align. '
                             f'Maybe the function is not scalar-valued?')


class _D_2order_(D):
    """
    Second-order numerical differentiation using central difference scheme.
    """

    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the second-order numerical derivative using central difference.

        Parameters:
        - X: torch.Tensor
            Input values where the derivative is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            The computed derivative values.
        """
        dX = self.eps
        return (self.func(X + dX, *args, **kwargs) - self.func(X - dX, *args, **kwargs)) / (2 * dX)


class _D_4order_(D):
    """
    Fourth-order numerical differentiation using a higher-order finite difference scheme.
    """

    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the fourth-order numerical derivative.

        Parameters:
        - X: torch.Tensor
            Input values where the derivative is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            The computed derivative values.
        """
        dX = self.eps

        return (
            (-1 / 12) * self.func(X + 2 * dX, *args, **kwargs) +
            (2 / 3) * self.func(X + dX, *args, **kwargs) -
            (2 / 3) * self.func(X - dX, *args, **kwargs) +
            (1 / 12) * self.func(X - 2 * dX, *args, **kwargs)
        ) / dX


class Grad(Diff):
    """
    Numerical gradient calculation of a function over the last dimension.

    This class acts as a factory returning an instance of the appropriate
    gradient calculation scheme based on the `order` parameter.

    Parameters:
    - func: Callable
        The function whose gradient is to be computed.
    - eps: float, optional
        Step size for numerical differentiation (default: 1e-5).
    - order: int, optional
        Order of the numerical differentiation scheme.
        Supported values: 1 (first-order), 2 (second-order), 4 (fourth-order).
        Default is first-order scheme.
    """

    def __new__(cls,
                func: Callable,
                eps: float = 1e-5,
                order: int = 1) -> 'Grad':
        if order == 1:
            instance = super().__new__(_Grad_1order_)
        elif order == 2:
            instance = super().__new__(_Grad_2order_)
        elif order == 4:
            instance = super().__new__(_Grad_4order_)
        else:
            raise ValueError(f"Invalid order: {order}. Supported orders are 1, 2, and 4.")
        return instance


    def __init__(self,
                 func: Callable,
                 eps: float = 1e-5,
                 order: int = 1):
        # Avoid reinitialization if already initialized by subclass
        if hasattr(self, 'func'):
            return
        super().__init__(func, eps)
        self.order = order


    def __check__(self, X):
        # TODO: implement this method
        pass


class _Grad_1order_(Grad):
    """
    First-order numerical gradient using central difference scheme.
    """

    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the gradient using first-order scheme (central difference).

        Parameters:
        - X: torch.Tensor
            Input tensor where the gradient is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            Gradient tensor of same shape as X.
        """
        outp = torch.zeros_like(X)
        dim = X.shape[-1]
        eye = torch.eye(dim, dtype=X.dtype, device=X.device)
        # Reshape eye for broadcasting: (1, 1, ..., dim, dim)
        shape_prefix = [1] * (X.ndim - 1)
        dX = self.eps * eye.view(*shape_prefix, dim, dim)

        for i in range(dim):
            # X shape: (..., dim)
            # dX[..., i] shape: (..., dim)
            outp[..., i] = (self.func(X + dX[..., i], *args, **kwargs) - self.func(X, *args, **kwargs)) / (self.eps)

        return outp


class _Grad_2order_(Grad):
    '''
    Second-order numerical gradient using central second-order scheme (central difference)
    '''
    
    def __call__(self, X, *args, **kwargs):

        outp = torch.zeros_like(X)
        dim = X.shape[-1]
        eye = torch.eye(dim, dtype=X.dtype, device=X.device)

        # Reshape eye for broadcasting: (1, 1, ..., dim, dim)
        shape_prefix = [1] * (X.ndim - 1)
        dX = self.eps * eye.view(*shape_prefix, dim, dim)

        for i in range(dim):
            # X shape: (..., dim)
            # dX[..., i] shape: (..., dim)
            outp[..., i] = (self.func(X + dX[..., i], *args, **kwargs) - self.func(X - dX[..., i], *args, **kwargs)) / (2 * self.eps)

        return outp


class _Grad_4order_(Grad):
    """
    Fourth-order numerical gradient using higher-order finite difference scheme.
    """

    def __call__(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the gradient using fourth-order scheme.

        Parameters:
        - X: torch.Tensor
            Input tensor where the gradient is computed.
        - *args, **kwargs:
            Additional arguments passed to the function.

        Returns:
        - torch.Tensor
            Gradient tensor of same shape as X.
        """
        outp = torch.zeros_like(X)
        dim = X.shape[-1]
        eye = torch.eye(dim, dtype=X.dtype, device=X.device)
        shape_prefix = [1] * (X.ndim - 1)
        dX = self.eps * eye.view(*shape_prefix, dim, dim)

        for i in range(dim):
            outp[..., i] = (
                (-1 / 12) * self.func(X + 2 * dX[..., i], *args, **kwargs) +
                (2 / 3) * self.func(X + dX[..., i], *args, **kwargs) -
                (2 / 3) * self.func(X - dX[..., i], *args, **kwargs) +
                (1 / 12) * self.func(X - 2 * dX[..., i], *args, **kwargs)
            ) / self.eps

        return outp


class Hessian(Diff):
    '''
    Class for calculating the Hessian matrix of a function
    '''

    def __init__(self,
                 func: callable,
                 eps: float = 1e-5, 
                 dim: int = 4
                 ):
        
        super().__init__(func, eps)


    def __call__(self, X, *args, **kwargs):

        hessian = torch.zeros(*X.shape[:-1], self.dim, self.dim)
        dVec = self.epseye.view(*[1 for _ in X.shape[:-1]], self.dim, self.dim)

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                f_ij = self.func(X + dVec[..., i] + dVec[..., j], *args, **kwargs)
                f_i_j = self.func(X + dVec[..., i] - dVec[..., j], *args, **kwargs)
                f_ij_ = self.func(X - dVec[..., i] + dVec[..., j], *args, **kwargs)
                f_i_j_ = self.func(X - dVec[..., i] - dVec[..., j], *args, **kwargs)
                hessian[..., i, j] = (f_ij - f_i_j - f_ij_ + f_i_j_) / (4 * self.eps ** 2)

        return hessian






