'''
This file describes an abstract class Particle, which holds routines

'''

from abc import ABC, abstractmethod
from bhtrace.geometry.spacetime.base import Spacetime

import torch


class Particle(ABC):
    """Abstract base class for all particle types.

    This class defines the interface for particles, including methods for
    calculating the Hamiltonian and its derivatives.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Particle:
            raise TypeError("Particle is an abstract class and cannot be instantiated directly. "
                            "Use a concrete subclass or the factory function `bhtrace.geometry.particle.create()`.")
        return super().__new__(cls)
 
    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the Particle instance.

        Args:
            spacetime (Spacetime): The spacetime in which the particle exists.
            **kwargs: Additional keyword arguments.
        """
        if spacetime is None:
            raise ValueError("A valid Spacetime object must be provided.")

        self.spacetime = spacetime
        self.__coords__ = spacetime.__coords__
        self.mu = None  # Particle mass
        self.r_max = torch.tensor([30.0])
        self.gtol = torch.tensor([1e-6, 1e6])
        self.color = None
        self.g_ = None 
        self.ginv_ = None
        self.dgX_ = None
        self.__name__ = self.__class__.__name__

    def __str__(self) -> str:
        return f'Particle: {self.color if self.color else self.__name__}'

    def state(self) -> dict:
        """Returns a dictionary representing the state of the particle.

        Returns:
            dict: A dictionary containing the particle's name, mass, color,
                  and spacetime state.
        """
        return {
            'name': self.__name__,
            'mu': self.mu,
            'color': self.color,
            'spacetime': self.spacetime.state()
        }

    @classmethod
    def from_dict(cls, state: dict) -> 'Particle':
        """Creates a Particle object from a state dictionary.

        Args:
            state (dict): A dictionary containing the particle's state.

        Returns:
            An instance of a `Particle` subclass.
        """
        from bhtrace.geometry.particle import create
        from bhtrace.geometry.spacetime import Spacetime
        state = state.copy()

        spacetime_state = state.pop('spacetime')
        spacetime = Spacetime.from_dict(spacetime_state)
        name = state.pop('name')
        return create(name=name, spacetime=spacetime, **state)

    @abstractmethod
    def Hmlt(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Calculates the particle's Hamiltonian.

        The Hamiltonian defines the dynamics of the particle in phase space.
        For a free particle, this is typically `0.5 * g^uv * P_u * P_v`.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].

        Returns:
            torch.Tensor: The Hamiltonian value at each point, shape [...].
        """
        return None

    @abstractmethod
    def energy(self, X: torch.Tensor, P: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Calculates the particle's energy as measured by an observer.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].
            u (torch.Tensor): Observer's contravariant 4-velocity `u^u`,
                              shape [..., 4].

        Returns:
            torch.Tensor: The measured energy, shape [...].
        """
        return None

    @abstractmethod
    def dHmlt(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Calculates the analytical partial derivatives of the Hamiltonian.

        This method should compute `dH/dX^p`.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].

        Returns:
            torch.Tensor: The Hamiltonian derivatives `dH/dX^p` at each point,
                          shape [..., 4].
        """
        return None

    def dHmlt_(self, X: torch.Tensor, P: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Numerically calculates the partial derivatives of the Hamiltonian.

        This method uses a second-order central difference scheme and serves as
        a general-purpose alternative to an analytical `dHmlt` method.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].
            eps (float, optional): The step size for the finite difference.
                                   Defaults to 1e-3.

        Returns:
            torch.Tensor: The Hamiltonian derivatives `dH/dX^p` at each point,
                          shape [..., 4].
        """

        dVec = eps*torch.eye(X.shape[-1]).view(*[1] * (X.ndim -1), 1, 1)

        # H = self.Hmlt(X, P)
        dH = torch.zeros_like(X)

        dH[..., 0] = (self.Hmlt(X+dVec[..., 0, :], P) - self.Hmlt(X-dVec[..., 0, :], P))/eps/2
        dH[..., 1] = (self.Hmlt(X+dVec[..., 1, :], P) - self.Hmlt(X-dVec[..., 1, :], P))/eps/2
        dH[..., 2] = (self.Hmlt(X+dVec[..., 2, :], P) - self.Hmlt(X-dVec[..., 2, :], P))/eps/2
        dH[..., 3] = (self.Hmlt(X+dVec[..., 3, :], P) - self.Hmlt(X-dVec[..., 3, :], P))/eps/2

        return dH


    @abstractmethod
    def normp(self, X, P):
        '''
        Method of normalizing particle impulse P at coord X
        ### Input:
        - X: torch.Tensor() - coordinate
        - P: torch.Tensor() - impulse

        ### Output:
        - P: torch.Tensor() - normalized impulse
        - mu: float - norm
        - v: float - spatial velocity
        '''
        return None


    def GetNullMomentum(self, X, v):
        '''
        Method for calculating covariant 4-impulse P_u for particle at point X with 3-velocity v.

        ### Inputs:
        - X: torch.Tensor [..., 4] - cooridnate
        - v: torch.Tensor [..., 4] - 4-velocity

        ### Outputs:
        - P: torch.Tensor() - initial impulse
        '''

        return NotImplementedError


    def GetDirection(self, X, P):
        '''
        Method for calculating direction of particle at point X with impulse P

        ### Inputs:
        - X: torch.Tensor() - coordinate
        - P: torch.Tensor() - impulse

        ### Outputs:
        - V: torch.Tensor() - 3-velocity

        '''

        return NotImplementedError


    def MomentumNorm(self, X, P):
        '''
        Method for calculating impulse norm (P^mu P_mu)

        ### Inputs:

        ### Outputs:


        '''

        return NotImplementedError


    def crit(self, X, P):
        '''
        Stopping condition

        ### Inputs:
        - X: torch.Tensor() - coordinate
        - P: torch.Tensor() - impulse

        ### Outputs:
        - bool
        '''

        detgX = torch.abs(torch.det(self.spacetime.g(X)))
        cr1 = torch.less(detgX, self.gtol[0])
        cr2 = torch.greater(detgX, self.gtol[1])

        # return False to continue
        return cr1 + cr2


class MockParticle(Particle):

    def __init__(self, spacetime: Spacetime, **kwargs):
        super().__init__(spacetime=spacetime, **kwargs)

    def Hmlt(self, X, P):
        return None

    def energy(self, X, P, u):
        return None

    def dHmlt(self, X, P):
        return None

    def normp(self, X, P):
        return None
