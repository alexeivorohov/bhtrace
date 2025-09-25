'''
This file contains an abstract class Spacetime, which calculates the principal
quantities associated with a given spacetime: the metric and the Levi-Civita connection
symbols. The abstract class Spacetime contains the following methods:

    g(self, X): 
        Abstract method for calculating the metric at a given point X
        (or at a batch of points, feature under development).

    ginv(self, X):
        Abstract method for calculating the inverse metric at a given point X
        (with raised indices).

    dg(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses the simplest first-oder
        difference scheme.

    dg_horder(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses order 2 or order 4 difference
        scheme for calculating derivatives.

    conn(self, X):
        Abstract method. Computes the Levi-Civita connection coefficients 
        (Christoffel symbols) via a method specified down the line.

    conn_(self, X, method='standard'):
        Computes the Levi-Civita connection coefficients numerically
        using either dg ('standard') or dg_horder ('horder') method. 

    crit(self, X):
        Abstract method. Computes the "proximity" value to 
        the singularity of the metric.
'''

from abc import ABC, abstractmethod
import inspect

import torch

from bhtrace.utils.linalg import tetrad_gd, tetrad_linalg
from bhtrace.utils import Cacher

# May be derivatives and connections should be moved to another class?
# In this case, dealing with different coordinate systems may be simpler (?)

class Spacetime(ABC):
    """Abstract base class for all spacetime geometries.

    This class defines the interface for spacetime metrics, including methods
    for calculating the metric tensor, its inverse, and connection coefficients.
    """
    __analytic_conn__ = False
    _g00_tol = -0.1
    cacher = Cacher()

    def __new__(cls, *args, **kwargs):
        if cls is Spacetime:
            raise TypeError("Spacetime is an abstract class and cannot be instantiated directly. "
                            "Use a concrete subclass or the factory function `bhtrace.geometry.spacetime.create()`.")
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Initializes the Spacetime instance.

        Note:
            Subclasses should call `super().__init__()`. Any arguments passed
            to the base class constructor are ignored.
        """
        pass

    def state(self) -> dict:
        """Returns a dictionary representing the state of the spacetime.

        Returns:
            dict: A dictionary containing the name of the spacetime class
                  and its parameters.
        """
        state = {'name': self.__class__.__name__}

        sig = inspect.signature(self.__class__.__init__)
        for param in sig.parameters.values():
            if param.name != 'self' and hasattr(self, param.name):
                attr = getattr(self, param.name)
                # To avoid serializing things that are not parameters
                if isinstance(attr, (int, float, str, bool)):
                     state[param.name] = attr
                elif isinstance(attr, torch.Tensor):
                        state[param.name] = attr.tolist()
        return state

    def horizon(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Function for determining apparent outer horizon (and, possibly, other horizons)
        
        Should be positive in outer space, zero on horizon and negative under the horizon
        '''
        return NotImplementedError                


    @classmethod
    def from_dict(cls, state: dict):
        """Creates a Spacetime object from a state dictionary.

        Args:
            state (dict): A dictionary containing the spacetime's name and parameters.

        Returns:
            An instance of a `Spacetime` subclass.
        """
        from bhtrace.geometry.spacetime import create
        name = state.pop('name')
        return create(name, **state)

    @cacher.attach
    def g(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates the metric tensor (covariant) at a given set of coordinates.

        Args:
            X (torch.Tensor): A tensor of shape [..., 4] representing the
                              spacetime coordinates.

        Returns:
            torch.Tensor: The metric tensor `g_uv` at each coordinate, with
                          shape [..., 4, 4].
        """
        pass

    @cacher.attach
    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates the inverse metric tensor (contravariant) at a given set of coordinates.

        Args:
            X (torch.Tensor): A tensor of shape [..., 4] representing the
                              spacetime coordinates.

        Returns:
            torch.Tensor: The inverse metric tensor `g^uv` at each coordinate,
                          with shape [..., 4, 4].
        """
        pass

    def detg(self, X):
        '''Determinant of the metric
        
        By default, torch linalg is used for evaluation of determinant.

        But for most metrics more optimal analytical expression is present and implemented.
        '''
        g = self.g(X)

        return torch.linalg.det(g)

    def dg(self, X: torch.Tensor, eps: float = 2e-5) -> torch.Tensor:
        """Numerically calculates the partial derivatives of the metric tensor.

        This method uses a first-order finite difference scheme.

        Args:
            X (torch.Tensor): A tensor of shape [..., 4] representing the
                              point(s) at which to evaluate the derivative.
            eps (float, optional): The step size for the finite difference.
                                   Defaults to 2e-5.

        Returns:
            torch.Tensor: The partial derivatives of the metric `d_p g_uv`
                          at each point, with shape [..., 4, 4, 4].
        """
        gX = self.g(X)
        dgX = torch.zeros(*X.shape[:-1], 4, 4, 4, device=X.device, dtype=X.dtype)

        dVec = torch.eye(4, device=X.device, dtype=X.dtype).repeat(*X.shape[:-1], 1, 1) * eps

        for i in range(4):
            dgX[..., i, :, :] = (self.g(X + dVec[..., i, :]) - gX) / eps

        return dgX

    def conn(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the Levi-Civita connection coefficients (Christoffel symbols).

        This method serves as a public interface, dispatching to the
        appropriate concrete implementation (analytical or numerical).

        Args:
            X (torch.Tensor): A tensor of shape [..., 4] representing the
                              evaluation point(s).

        Returns:
            torch.Tensor: The connection symbols `Gamma^p_uv` at each point,
                          with shape [..., 4, 4, 4]. The first index is
                          contravariant, the other two are covariant.
        """
        return self.conn_(X, method='standard')

    def conn_(self, X: torch.Tensor, method: str = 'standard') -> torch.Tensor:
        """Numerically evaluates connection symbols from metric derivatives.

        Args:
            X (torch.Tensor): A tensor of shape [..., 4] representing the
                              evaluation point(s).
            method (str, optional): The numerical method to use. Currently,
                                    only 'standard' is implemented. Defaults to 'standard'.

        Returns:
            torch.Tensor: The connection symbols `Gamma^p_uv` at each point,
                          with shape [..., 4, 4, 4].
        """
        if method == 'standard':
            g_duv = self.dg(X)
            ginv_ = self.ginv(X)
            
        elif method == 'horder':
            # g_duv = self.dg_horder(X) # Not implemented
            raise NotImplementedError("High-order derivative method not implemented.")
            ginv_ = self.ginv(X)

        dg0 = torch.einsum('...md, ...duv ->...muv', ginv_, g_duv)
        dg1 = torch.einsum('...mv,...duv -> ...mdv', ginv_, g_duv)
        dg2 = torch.einsum('...mu, ...duv -> ...mud', ginv_, g_duv)
        
        return 0.5*( - dg0 + dg1 + dg2)
    
    def tetrad(self, X: torch.Tensor, method: str = 'gd'):
        """Computes the tetrad frame. (Not fully implemented)."""
        if method == 'gd':
            return tetrad_gd(self, X)
        else:
            return tetrad_linalg(self, X)
   
    def compile(self):
        """Compiles the class with `torch.jit.script` for performance.
        
        Note:
            This is experimental and may not work for all subclasses.
        """
        return torch.jit.script(self)

    def __str__(self) -> str:
        return self.__class__.__name__
    

class MockSpacetime(Spacetime):
    
    def __init__(self, coefs=[1.0, 2.0, 3.0, 5.0]):
        '''
        :class:`Spacetime()` implementation, used for test purposes.
        '''
        super().__init__()
        self.coefs = torch.tensor(coefs)
        pass


    def g(self, X):
        
        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - self.coefs[0]
        outp[..., 1, 1] = self.coefs[1]
        outp[..., 2, 2] = self.coefs[2]
        outp[..., 3, 3] = self.coefs[3]

        return outp


    def ginv(self, X):

        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - 1/self.coefs[0]
        outp[..., 1, 1] = 1/self.coefs[1]
        outp[..., 2, 2] = 1/self.coefs[2]
        outp[..., 3, 3] = 1/self.coefs[3]

        return outp