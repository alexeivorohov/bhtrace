from typing import Tuple

import torch

from bhtrace.medium._base import Medium, MEDIUM_REGISTRY
from bhtrace.geometry.spacetime import Spacetime

@MEDIUM_REGISTRY.register("mock")
class MockMedium(Medium):
    """
    Primitive medium for test purposes with all properties constant
    """

    def __init__(self, flux: float = 1.0, temp: float = 1.0, density: float = 1.0, opacity_val: float = 1.0, r: float = 5.0, spacetime: Spacetime = None):
        super().__init__(spacetime)
        self._temp = torch.tensor(temp)
        self._density = torch.tensor(density)
        self._opacity = torch.tensor(opacity_val)
        self._flux = torch.tensor(flux)
        self._r = torch.tensor(r)

    def surface_flux(self, x):
        return self._flux*torch.ones_like(x[..., 0])

    def temperature(self, x):
        return self._temp*torch.ones_like(x[..., 0])

    def rest_mass_density(self, x):
        return self._density*torch.ones_like(x[..., 0])

    def opacity(self, x):
        return self._opacity*torch.ones_like(x[..., 0])
    
    def signed_distance(self, x):
        # r is the second coordinate in spherical
        return x[..., 1] - self._r
    
    def hit_condition(self, s0, s1):
        return torch.sign(s0) != torch.sign(s1)


@MEDIUM_REGISTRY.register("volumetricshell")
class VolumetricShell(Medium):
    def __init__(self, spacetime: Spacetime, r_in: float = 6.0, r_out: float = 20.0, omega: float = 0.0):
        super().__init__(spacetime)
        self.r_in = r_in
        self.r_out = r_out
        self.omega = omega

    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        # This defines a spherical shell between r_in and r_out, with negative values inside.
        return torch.max(r - self.r_out, self.r_in - r)

    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        # A hit occurs if the new point is inside the medium.
        return s1 <= 0

    def adjust_hit(self, x0: torch.Tensor, x1: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple linear interpolation to find the boundary crossing point
        # This is more accurate than just taking the midpoint.
        t = s0 / (s0 - s1)
        x_hit = x0 + t.unsqueeze(-1) * (x1 - x0)
        p_hit = p0 + t.unsqueeze(-1) * (p1 - p0)
        return x_hit, p_hit

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        # Stationary fluid in the coordinate frame.
        v_phi = x[..., 1] * self.omega
        u = torch.zeros_like(x)
        u[..., 0] = (1 - v_phi.pow(2)).rsqrt()
        u[..., 3] = v_phi
        return u

    # Methods required by the radiative models
    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 1e4)

    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 1.0)

    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 1.0)

    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 10.0)
