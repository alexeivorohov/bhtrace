"""
Abstract Base Classes for Spacetime Geometries.

This module defines the abstract base class `Spacetime` for representing
various spacetime geometries. It provides a common interface for calculating
essential geometric quantities such as the metric tensor, its inverse,
and connection coefficients.

"""

from abc import ABC, abstractmethod
from typing import Literal
import inspect

import torch

from bhtrace.utils.numrel import numeric_conn, numeric_tetrad, jacobian
from bhtrace.utils import Cacher, Logger, LOG, Registry


class Spacetime(ABC):
    """Abstract base class for all spacetime geometries.

    This class defines the interface for spacetime metrics, including methods
    for calculating the metric tensor, its inverse, and connection coefficients.

    Attributes
    ----------
    _has_analytic_conn : bool
        Flag indicating if analytic expression for connection coefficients provided.
    _has_analytic_tetrad : bool
        Flag indicating if analytic expression for tetrads provided.
    _coords : str or None
        Name of the coordinate system used (e.g., "Spherical").
    _eps : float
        Step size for numerical differentiation.
    _diff_ord : int
        Order of the numerical differentiation scheme.
    _num_tetrad_scheme : str
        Scheme used for numerical tetrad evaluation.
        .. note:: The available schemes are not explicitly defined, making
                  this property's options unclear without further context.
    cacher : Cacher
        Cache controller instance. See `bhtrace.utils.cacher`.
    r_h : float
        Radius of the apparent horizon.

    """

    _has_analytic_conn: str = False
    """True if analytic connections are provided"""
    _has_analytic_tetrad: str = False
    """True if analytic tetrad are provided"""
    _coords: str = None
    """Coordinate system of the spacetime"""
    _eps: float = 1e-5
    """step size for numerical differentiation"""
    _diff_ord: int = 2
    """order of numerical differentiation scheme"""
    _num_tetrad_scheme: str = "gd"
    """numerical tetrad evaluation scheme"""

    cacher: Cacher = Cacher()
    """Cache controller, see bhtrace.utils.cacher"""

    r_h: float = 2.0
    """Apparent horizon radius"""

    def __new__(cls, *args, **kwargs):
        if cls is Spacetime:
            raise TypeError(
                "Spacetime is an abstract class and cannot be instantiated directly. "
                "Use a concrete subclass or the factory function `bhtrace.geometry.spacetime.create()`."
            )
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Initializes the Spacetime instance.

        Notes
        -----
        Subclasses should call `super().__init__()`. Any arguments passed
        to the base class constructor are ignored.
        """
        pass

    def state(self) -> dict:
        """Return a dictionary representing the state of the spacetime.

        Returns
        -------
        dict
            A dictionary containing the name of the spacetime class and its
            initialization parameters.
        """
        state = {"name": self.__class__.__name__}

        sig = inspect.signature(self.__class__.__init__)
        for param in sig.parameters.values():
            if param.name != "self" and hasattr(self, param.name):
                attr = getattr(self, param.name)
                # To avoid serializing things that are not parameters
                if isinstance(attr, (int, float, str, bool)):
                    state[param.name] = attr
                elif isinstance(attr, torch.Tensor):
                    state[param.name] = attr.tolist()
        return state

    def horizon(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the horizon function.

        This function determines the location of horizons. It should be
        positive outside the horizon, zero on the horizon, and negative
        inside it.

        .. note:: This method is abstract and must be implemented by subclasses.

        Parameters
        ----------
        X : torch.Tensor
            Spacetime coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Value of the horizon function at the given coordinates.
        """
        return NotImplementedError

    @classmethod
    def from_dict(cls, state: dict):
        """Create a Spacetime object from a state dictionary.

        Parameters
        ----------
        state : dict
            A dictionary containing the spacetime's name and parameters.

        Returns
        -------
        Spacetime
            An instance of a `Spacetime` subclass.
        """
        from bhtrace.geometry.spacetime import SPACETIME_REGISTRY

        name = state.pop("name")
        return SPACETIME_REGISTRY.create(name, **state)

    @cacher.attach
    def g(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate the metric tensor (covariant).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing the spacetime coordinates.

        Returns
        -------
        torch.Tensor
            The metric tensor `g_uv` at each coordinate, with shape [..., 4, 4].
        """
        pass

    @cacher.attach
    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate the inverse metric tensor (contravariant).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing the spacetime coordinates.

        Returns
        -------
        torch.Tensor
            The inverse metric tensor `g^uv` at each coordinate, with
            shape [..., 4, 4].
        """
        pass

    def detg(self, X: torch.Tensor):
        """Calculate the determinant of the metric.

        By default, `torch.linalg.det` is used. For many metrics, a more
        optimal analytical expression is implemented in the subclass.

        Parameters
        ----------
        X : torch.Tensor
            Spacetime coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The determinant of the metric at each point.
        """
        g = self.g(X)

        return torch.linalg.det(g)

    def dg(self, X: torch.Tensor) -> torch.Tensor:
        """Numerically calculate the partial derivatives of the metric tensor.

        This method uses a finite difference scheme. The order is controlled
        by the `_diff_ord` attribute.

        Parameters
        ----------
        X : torch.Tensor
            Point(s) of shape [..., 4] at which to evaluate the derivative.

        Returns
        -------
        torch.Tensor
            The partial derivatives of the metric `d_p g_uv` at each point,
            with shape [..., 4, 4, 4].
        """
        return jacobian(self.g, X, eps=self._eps, order=self._diff_ord)

    def conn(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the Levi-Civita connection coefficients (Christoffel symbols).

        This method serves as a public interface, dispatching to the
        appropriate concrete implementation (analytical or numerical).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing the evaluation point(s).

        Returns
        -------
        torch.Tensor
            The connection symbols `Gamma^p_uv` at each point, with shape
            [..., 4, 4, 4]. The first index is contravariant.
        """
        return self.conn_(X)

    def conn_(self, X: torch.Tensor) -> torch.Tensor:
        """Numerically evaluate connection symbols from metric derivatives.

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing the evaluation point(s).

        Returns
        -------
        torch.Tensor
            The connection symbols `Gamma^p_uv` at each point, with shape
            [..., 4, 4, 4].
        """
        return numeric_conn(self.g, X, eps=self._eps, order=self._diff_ord)

    def tetrad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute the local tetrad.

        This method should be overridden in subclasses for analytic evaluation.
        By default, it calls the numerical implementation `tetrad_`.

        Parameters
        ----------
        x : torch.Tensor
            Batch of points of shape [..., 4] for which to compute the tetrad.
        *args, **kwargs
            Additional, metric-specific parameters for the tetrad.

        Returns
        -------
        torch.Tensor
            A collection of tetrad vectors of shape [..., 4, 4].
        """
        return self.tetrad_(x)

    def tetrad_(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate the local tetrad numerically.

        Parameters
        ----------
        x : torch.Tensor
            Batch of points of shape [..., 4] for which to compute the tetrad.
        **kwargs
            Additional parameters for `bhtrace.utils.numrel.numeric_tetrad`.

        Returns
        -------
        torch.Tensor
            A collection of tetrad vectors of shape [..., 4, 4].

        """
        # TODO: Allow option to apply boosts/setup velocity of new frame
        return numeric_tetrad(self.g, x, method=self._num_tetrad_scheme, **kwargs)

    def lnrf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the tetrad for a locally non-rotating frame (LNRF).

        Notes
        -----
        For stationary spacetimes, this typically returns the local tetrad
        with default parameters. For other spacetimes, a specific
        implementation may be required.

        Parameters
        ----------
        x : torch.Tensor
            Batch of points of shape [..., 4] for which to compute the tetrad.

        Returns
        -------
        torch.Tensor
            A collection of tetrad vectors of shape [..., 4, 4].
        """
        return self.tetrad(x)

    def r_isco(self) -> float:
        """Return the radius of the innermost stable circular orbit (ISCO).

        .. note:: This is a placeholder and needs a proper implementation.

        Returns
        -------
        float
            The ISCO radius for the equatorial plane.
        """
        # TODO: implement this method.
        return 6.0

    def compile(self):
        """Compile the class with `torch.jit.script` for performance.

        .. note:: This is an experimental feature and may not work for all
                  subclasses.

        """
        return torch.jit.script(self)

    @property
    def real(self) -> "Spacetime":
        """Return the real spacetime.

        Notes
        -----
        The purpose of this property is unclear without more context on
        effective spacetime geometries. It returns the object itself.
        """
        return self

    @property
    def eff(self) -> "Spacetime":
        """Return the effective spacetime.

        Notes
        -----
        The purpose of this property is unclear without more context on
        effective spacetime geometries. It returns the object itself.
        """
        return self

    def __repr__(self) -> str:
        return self.__class__.__name__


SPACETIME_REGISTRY = Registry(Spacetime)


@SPACETIME_REGISTRY.register('mock')
class MockSpacetime(Spacetime):
    """
    A mock `Spacetime` implementation for testing purposes.

    This class implements a simple diagonal metric with constant coefficients.

    Parameters
    ----------
    coefs : list of float, optional
        Coefficients for the diagonal metric components, by default
        [1.0, 2.0, 3.0, 5.0].
    """

    def __init__(self, coefs=[1.0, 2.0, 3.0, 5.0]):
        super().__init__()
        self.coefs = torch.tensor(coefs)
        pass

    def g(self, X):
        """Calculate the mock metric tensor."""
        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = -self.coefs[0]
        outp[..., 1, 1] = self.coefs[1]
        outp[..., 2, 2] = self.coefs[2]
        outp[..., 3, 3] = self.coefs[3]

        return outp

    def ginv(self, X):
        """Calculate the inverse mock metric tensor."""
        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = -1 / self.coefs[0]
        outp[..., 1, 1] = 1 / self.coefs[1]
        outp[..., 2, 2] = 1 / self.coefs[2]
        outp[..., 3, 3] = 1 / self.coefs[3]

        return outp
