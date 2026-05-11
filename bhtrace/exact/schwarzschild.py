from typing import Tuple, List

import torch

from ._base import AnalyticSolution
from bhtrace.trajectory import Trajectory
from bhtrace.geometry.spacetime import SchwSchild

class Schwarzschild(AnalyticSolution):
    """
    Analytic solution for geodesics in Schwarzschild spacetime.
    
    Uses elliptic functions for trajectory calculation.

    The procedure is as follows:
    1. For given initial conditions, calculate the invariants of motion (energy and angular momentum).
    2. Use the invariants to determine the type of trajectory (bound, unbound) and the parameters of the elliptic functions.


    References
    ----------

    """

    def __init__(self, particle_mass: float = 0.0):
        self.particle_mass = particle_mass
        self.m2 = particle_mass ** 2
        self.mu = 0 if particle_mass == 0 else 1


    def forward(self, x: torch.Tensor, v: torch.Tensor, tspan: torch.Tensor) -> Trajectory:
        """
        Calculate geodesics of particle in Schwarzschild spacetime analytically.
        
        Parameters
        ----------
        x : torch.Tensor
            Initial position of particle (4-vector).
        v : torch.Tensor
            Initial velocity of particle (4-vector).
        tspan : torch.Tensor
            Time span for trajectory calculation.

        Returns
        -------
        Trajectory
            Trajectory of particle in Schwarzschild spacetime.
        """
        ...

    def to_dimensionless(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def from_dimensionless(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


    def mino_time(self, s: torch.Tensor) -> torch.Tensor:
        ...

    def motion_invariants(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor]:
        energy = ...
        angular_momentum = ...
        return energy, angular_momentum
    
    def quartic_coefficients(
        self, 
        energy: torch.Tensor, 
        angular_momentum: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        a0 = energy**2 - self.mu
        a1 = 0.5 * self.mu * torch.ones_like(a0)
        a2 = - angular_momentum**2 / 6
        a3 = a2 * 3
        a4 = torch.zeros_like(a0)
        return a0, a1, a2, a3, a4

    def quartic_invariants(
        self, 
        a0: torch.Tensor,
        a1: torch.Tensor, 
        a2: torch.Tensor, 
        a3: torch.Tensor,
        a4: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        g1 = a0*a4 - 4*a1*a3 + 3*a2**2
        g2 = a0*a2*a4 + 2*a1*a2*a3 - a2**3 - a0*a3**2 - a1**2*a4
        return g1, g2


