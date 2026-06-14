"""
This module defines the `VolumetricShell` class, an implementation of a
spherically symmetric medium representing a volumetric shell. This model is
useful for simulating astrophysical objects like stellar envelopes or gas clouds
that are extended in three dimensions.

The `VolumetricShell` provides constant values for density, opacity, temperature,
and surface flux within its defined radial boundaries, and supports a
stationary or rotating fluid velocity field.
"""

from typing import Tuple

import torch

from bhtrace.medium._base import Medium, MEDIUM_REGISTRY
from bhtrace.geometry.spacetime import Spacetime
import bhtrace.utils.units as bhU


@MEDIUM_REGISTRY.register("volumetricshell")
class VolumetricShell(Medium):
    """
    Represents a spherically symmetric volumetric shell of matter in spacetime.

    This medium is characterized by inner and outer radii, a constant density,
    opacity, and temperature within its boundaries, and a uniform rotation rate.

    Attributes
    ----------
    r_in : float
        Inner radius of the spherical shell in geometric units.
    r_out : float
        Outer radius of the spherical shell in geometric units.
    omega : float
        Angular velocity of the shell's fluid in geometric units.
    _density : float
        Constant rest mass density of the shell in geometric units.
    _opacity : float
        Constant opacity of the shell in geometric units.
    _flux : float
        Constant surface flux of the shell in geometric units, derived from
        the characteristic temperature.

    Methods
    -------
    signed_distance(x)
        Calculates the signed distance to the shell's boundaries.
    hit_condition(s0, s1)
        Determines if a trajectory segment has entered the shell.
    flux_density(x)
        Calculates the radiation flux density.
    adjust_hit(x0, x1, p0, p1, s0, s1)
        Adjusts hit coordinates (returns `x0, p0` for volumetric objects).
    velocity(x)
        Calculates the 4-velocity of the shell's fluid.
    temperature(x)
        Calculates the temperature within the shell.
    rest_mass_density(x)
        Calculates the rest mass density within the shell.
    opacity(x)
        Calculates the opacity within the shell.
    surface_flux(x)
        Calculates the surface flux within the shell.
    """

    def __init__(
        self,
        spacetime: Spacetime,
        mass: float = 1.0,
        temperature=1e6,
        density: float = 1e-3,
        opacity: float = 5e-3,
        r_in: float = 6.0,
        r_out: float = 20.0,
        omega: float = 0.0,
    ):
        """
        Initializes a VolumetricShell medium.

        Parameters
        ----------
        spacetime : bhtrace.geometry.spacetime.Spacetime
            The spacetime geometry in which the shell exists.
        mass : float, optional
            Characteristic mass of the central object for unit system setup
            in SI units (kg). Defaults to 1.0 kg.
        temperature : float, optional
            Characteristic temperature of the shell in SI units (K).
            Defaults to 1e6 K.
        density : float, optional
            Constant rest mass density of the shell material in SI units
            (:math:`kg/m^3`). Defaults to 1e-3 :math:`kg/m^3`.
        opacity : float, optional
            Constant opacity of the shell material in SI units
            (:math:`m^2/kg`). Defaults to 5e-3 :math:`m^2/kg`.
        r_in : float, optional
            Inner radius of the spherical shell in geometric units.
            Defaults to 6.0.
        r_out : float, optional
            Outer radius of the spherical shell in geometric units.
            Defaults to 20.0.
        omega : float, optional
            Angular velocity of the shell's fluid (about the z-axis) in
            geometric units. Defaults to 0.0 (stationary).

        Attributes
        ----------
        _density : float
            Internal constant density converted to geometric units.
        _opacity : float
            Internal constant opacity converted to geometric units.
        _flux : float
            Internal constant surface flux converted to geometric units,
            calculated from the `temperature`.

        Notes
        -----
        The `mass` and `temperature` parameters are primarily used to set up
        the `GRRTUnitSystem` in the base `Medium` class.
        """
        super().__init__(spacetime=spacetime, mass=mass, temperature=temperature)
        self.r_in = r_in
        self.r_out = r_out
        self.omega = omega
        self._density = (density * bhU.density).to(self.units).f
        self._opacity = (opacity * bhU.area / bhU.mass).to(self.units).f
        self._flux = (temperature**4 * bhU.K * bhU.sigma_SB).to(self.units).f

    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        # This defines a spherical shell between r_in and r_out, with negative values inside.
        # Max(r - r_out, r_in - r) gives negative when r_in < r < r_out, and positive otherwise.
        return torch.max(r - self.r_out, self.r_in - r)

    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        # A hit occurs if the new point is inside the medium (signed distance <= 0).
        return s1 <= 0

    def flux_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.surface_flux(x)

    def adjust_hit(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        s0: torch.Tensor,
        s1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For volumetric shells, the entry point into the volume is usually considered the "hit"
        # rather than an exact boundary intersection. So we return the starting point x0, p0.
        """
        return x0, p0

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        # Assume purely azimuthal motion for the shell
        r = x[..., 1]
        v_phi = (
            self.omega * r
        )  # Simple linear velocity in coordinate system for constant omega
        u = torch.zeros_like(x)
        u[..., 0] = (1 - v_phi).rsqrt()
        u[..., 3] = v_phi
        return u

    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 1.0)

    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], self._density)

    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], self._opacity)

    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], self._flux)

    def pressure(self, x: torch.Tensor):
        return torch.ones_like(x[..., 0])
