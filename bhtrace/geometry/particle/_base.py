"""
This file describes an abstract class Particle

"""
from abc import ABC, abstractmethod

import torch

from bhtrace.geometry.spacetime._base import Spacetime
from bhtrace.utils import Cacher, Registry
from bhtrace.utils.diff import jacobian


class Particle(ABC):
    """Abstract base class for all particle types.

    This class defines the interface for particles, including methods for
    calculating the Hamiltonian and its derivatives.

    Attributes
    ----------
    cacher : Cacher
        An instance of the Cacher utility for memoization.
    spacetime : Spacetime
        The spacetime environment in which the particle exists.
    mu : float or None
        The mass of the particle.
    r_max : torch.Tensor
        The maximum distance from the coordinate center within which the particle exists.
    g_ : torch.Tensor or None
        The metric tensor, populated by spacetime methods.
    ginv_ : torch.Tensor or None
        The inverse metric tensor, populated by spacetime methods.
    dgX_ : torch.Tensor or None
        Derivatives of the metric tensor, populated by spacetime methods.
    """

    cacher = Cacher()
    _eps: float
    _diff_ord: int

    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the Particle instance.

        Parameters
        ----------
        spacetime : Spacetime
            The spacetime in which the particle exists.
        **kwargs
            Additional keyword arguments.
        """
        if spacetime is None:
            raise ValueError("A valid Spacetime object must be provided.")

        self.spacetime = spacetime
        """Spacetime in which the particle exists"""
        self._eps = spacetime._eps
        self._diff_ord = spacetime._diff_ord

        self.__coords__ = spacetime._coords
        """Coordinates of the particle """

        self.mu = None
        """Particle mass"""

        self.r_max = torch.tensor([30.0])
        """Maximal distance from coordinate centrer, within particle exists"""

        self.g_ = None
        self.ginv_ = None
        self.dgX_ = None
        self.__name__ = self.__class__.__name__

    def state(self) -> dict:
        """Returns a dictionary representing the state of the particle.

        Returns
        -------
        dict
            A dictionary containing the particle's name, mass,
            and spacetime state.
        """
        return {
            "name": self.__name__,
            "mu": self.mu,
            "spacetime": self.spacetime.state(),
        }

    # @classmethod
    # def from_dict(cls, state: dict) -> "Particle":
    #     """Creates a Particle object from a state dictionary.

    #     Parameters
    #     ----------
    #     state : dict
    #         A dictionary containing the particle's state.

    #     Returns
    #     -------
    #     Particle
    #         An instance of a `Particle` subclass.
    #     """
    #     from bhtrace.geometry.particle import create
    #     from bhtrace.geometry.spacetime import Spacetime

    #     state = state.copy()

    #     spacetime_state = state.pop("spacetime")
    #     spacetime = Spacetime.from_dict(spacetime_state)
    #     name = state.pop("name")
    #     return create(name=name, spacetime=spacetime, **state)

    # @cacher.attach
    def hmlt(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculates the particle's Hamiltonian.

        The Hamiltonian defines the dynamics of the particle in phase space.
        For a free particle, this is typically `0.5 * g^uv * P_u * P_v`.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates `x^q` of shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u` of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The Hamiltonian value at each point, shape [...].
        """
        raise NotImplementedError

    def _hmlt_px(self, p: torch.Tensor, x: torch.Tensor):
        """Helper function for differentiating hamiltonian w.r.t. p"""
        return self.hmlt(x, p)

    @abstractmethod
    def energy(self, x: torch.Tensor, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Calculates the particle's energy as measured by an observer.

        Parameters
        ----------
        x : torch.Tensor
            Spacetime coordinates, shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `P_u`, shape [..., 4].
        u : torch.Tensor
            Observer's contravariant 4-velocity `u^u`, shape [..., 4].

        Returns
        -------
        torch.Tensor
            The measured energy, shape [...].
        """
        return None

    # @cacher.attach
    def dx_hmlt(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculates the partial derivatives of the Hamiltonian w.r.t. position.

        If not overridden by an analytical expression in a particle implementation,
        this method returns **numerical** derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates of shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u` of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Hamiltonian partial derivatives `dH/dx` at each point, shape [..., 4].
        """
        return self.dx_hmlt_(x, p)

    # @cacher.attach
    def dx_hmlt_(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Numerically calculates the partial derivatives of the Hamiltonian w.r.t. position.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates of shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u` of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Hamiltonian partial derivatives `dH/dx` at each point, shape [..., 4].
        """
        return jacobian(
            self.hmlt, x, eps=self._eps, order=self._diff_ord, kwargs={"p": p}
        )

    # @cacher.attach
    def dp_hmlt(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculates the partial derivatives of the Hamiltonian w.r.t. momentum.

        If not overridden by an analytical expression in a particle implementation,
        this method returns **numerical** derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates of shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u` of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Hamiltonian partial derivatives `dH/dp` at each point, shape [..., 4].
        """
        return self.dp_hmlt_(x, p)

    # @cacher.attach
    def dp_hmlt_(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Numerically calculates the partial derivatives of the Hamiltonian w.r.t. momentum.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates of shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u` of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Hamiltonian partial derivatives `dH/dp` at each point, shape [..., 4].
        """
        return jacobian(
            self._hmlt_px, p, eps=self._eps, order=self._diff_ord, kwargs={"x": x}
        )

    # @abstractmethod
    # def normp(self, X, P):
    #     """Normalizes the particle's 4-momentum.

    #     Parameters
    #     ----------
    #     X : torch.Tensor
    #         Particle coordinates.
    #     P : torch.Tensor
    #         Particle 4-momentum.

    #     Returns
    #     -------
    #     P_normalized : torch.Tensor
    #         Normalized 4-momentum.
    #     mu : float
    #         Norm of the momentum.
    #     v : float
    #         Spatial velocity.
    #     """
    #     return None

    def null_momentum(self, x: torch.Tensor, v: torch.Tensor):
        """Calculates the covariant 4-momentum `P_u` for a particle.

        Calculates `P_u` for a particle at a given point `X` with a given
        3-velocity `v`.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates, shape [..., 4].
        v : torch.Tensor
            4-velocity, shape [..., 4]. The spatial part is used.

        Returns
        -------
        torch.Tensor
            The initial 4-momentum `P_u`.
        """
        return NotImplementedError

    def spatial_velocity(self, x: torch.Tensor, p: torch.Tensor):
        """Calculates the spatial velocity of a particle.

        Parameters
        ----------
        X : torch.Tensor
            Particle coordinates.
        P : torch.Tensor
            Particle 4-momentum.

        Returns
        -------
        torch.Tensor
            The particle's 3-velocity.
        """
        return NotImplementedError

    def momentum_norm(self, x: torch.Tensor, p: torch.Tensor):
        """Calculates the norm of the 4-momentum, `P^mu * P_mu`.

        Parameters
        ----------
        X : torch.Tensor
            Particle coordinates.
        P : torch.Tensor
            Particle 4-momentum.

        Returns
        -------
        NotImplementedError
        """
        return NotImplementedError

    def crit(self, x: torch.Tensor, p: torch.Tensor):
        """Checks the stopping condition for the particle's trajectory.

        The condition is based on the determinant of the metric `g`.

        Parameters
        ----------
        X : torch.Tensor
            Particle coordinates.
        P : torch.Tensor
            Particle 4-momentum.

        Returns
        -------
        bool
            True if the stopping condition is met, False otherwise.
        """
        detgX = torch.abs(torch.det(self.spacetime.g(x)))
        cr1 = torch.less(detgX, self.gtol[0])
        cr2 = torch.greater(detgX, self.gtol[1])

        # return False to continue
        return cr1 + cr2

PARTICLE_REGISTRY = Registry(Particle)


@PARTICLE_REGISTRY.register('mock')
class MockParticle(Particle):
    """A particle with no dynamics, for testing and placeholder purposes."""

    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the MockParticle instance."""
        super().__init__(spacetime=spacetime, **kwargs)

    def hmlt(self, X, P):
        """Hamiltonian for a mock particle (returns None)."""
        return None

    def energy(self, X, P, u):
        """Energy for a mock particle (returns None)."""
        return None

    def dx_hmlt(self, X, P):
        """Hamiltonian derivative for a mock particle (returns None)."""
        return None

    def normp(self, X, P):
        """Momentum normalization for a mock particle (returns None)."""
        return None
