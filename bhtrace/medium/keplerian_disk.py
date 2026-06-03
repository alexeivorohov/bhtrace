r"""
This module provides the `KeplerianDisk` class, a simplest concrete implementation of
a geometrically thin accretion disk following .


"""

from typing import Tuple, Optional

import math
import torch

from bhtrace.medium._base import Medium, MEDIUM_REGISTRY
from bhtrace.geometry.spacetime import Spacetime
from bhtrace.medium._flavours import ThinDisk
import bhtrace.utils.units as bhU

@MEDIUM_REGISTRY.register("keplerian_disk")
class KeplerianDisk(ThinDisk):
    r"""
    Physically motivated Keplerian thin disk in geometric units.

    This implementation largely follows the conventions described in [1]_.

    Attributes
    ----------
    mass : float
        Mass of the parent object in SI units
    mass_dot : float
        Accretion rate in SI units
    alpha : float
        Shakura-Sunyaev Alpha-viscosity parameter (dimensionless).
    mu : float
        Mean molecular mass of gas in units of proton mass [:math:`m_p`].
    kappa_es : float
        Thomson electron-scattering opacity in SI units [:math:`m^2 kg^{-1}`].
    r_cut : float
        Disk cutoff radius in units of Schwarzschild radii [:math:`R_s`].
    r_in : float
        Inner radius of the disk, typically the ISCO radius.
    

    References
    ----------
    .. [1] Armitage, P. J. (2022). Lecture notes on accretion disk physics. arXiv preprint arXiv:2201.07262.
    """
    def __init__(
        self,
        spacetime: Spacetime,
        mass: float = 1.00,
        mass_dot: float = 1e-2,
        alpha: float = 0.05,
        kappa_es: float = 0.005,
        mu: float = 2.0,
        r_cut: float = 30,
        clockwise: bool = False
    ):
        r"""
        Initializes a Keplerian thin disk model.

        Parameters
        ----------
        spacetime : bhtrace.geometry.spacetime.Spacetime
            The spacetime geometry in which the disk exists.
        mass : float, optional
            Mass of the parent object in units of solar mass [:math:`M_{\odot}`].
            Defaults to 1.0 :math:`M_{\odot}`.
        mass_dot : float, optional
            Accretion rate in units of solar mass per year [:math:`M_{\odot} yr^{-1}`].
            Defaults to 1e-2 :math:`M_{\odot} yr^{-1}`.
        alpha : float, optional
            Shakura-Synyaev Alpha-viscosity parameter (dimensionless).
            Defaults to 0.05.
        kappa_es : float, optional
            Thomson electron-scattering opacity in SI units [:math:`m^2 kg^{-1}`].
            Defaults to 0.005 :math:`m^2 kg^{-1}`.
        mu : float, optional
            Mean molecular mass of gas in units of proton mass [:math:`m_p`].
            Defaults to 2.0.
        r_cut : float, optional
            Disk cutoff radius in units of Schwarzschild radii [:math:`R_s`].
            Defaults to 30 :math:`R_s`.
        clockwise : bool, optional
            If True, the disk is assumed to rotate clockwise. Otherwise,
            counter-clockwise. Defaults to False.
        
        Notes
        -----
        This constructor performs unit conversions to set up internal scaling
        factors based on the provided SI units for mass, accretion rate, and
        opacity, allowing the derived quantities to be consistent with the
        geometric unit system.
        """
        self.mass = mass
        self.r_in = 6.0 # TODO: Determine by `spacetime`
        self.mu = mu
        self.alpha = alpha
        self.kappa_es = kappa_es 
        
        # Unit conversions for internal scaling factors
        mass_si = mass * bhU.m_sun
        mass_dot_si = mass_dot * bhU.m_sun / bhU.year
        R_s = 2 * bhU.G * mass_si / bhU.c**2
        
        # Characteristic scales derived from physical parameters
        sf_scale = 3 * bhU.G * mass_si * mass_dot_si / 8 / math.pi  / R_s**3
        t_char = (sf_scale / bhU.sigma_SB).pow(0.25)
        cs_scale = (bhU.kB / mu / bhU.m_p * t_char).sqrt()
        fd_scale = 3 * mass_dot_si / 64 / math.pi / cs_scale # This needs re-evaluation of its derivation, looks simplified.
        sd_scale = (mass_dot_si / 3 / math.pi )

        super().__init__(
            spacetime=spacetime, 
            r_cut=r_cut, 
            clockwise=clockwise,
            mass=mass_si.si, # Pass SI mass to base Medium for unit system setup
            temperature=t_char.si # Pass SI temperature to base Medium for unit system setup
        )
        
        # Store internal scaling factors in geometric units
        self._sf_scale = sf_scale.to(self.units).f # Surface flux scale in geometric units
        self._cs_scale = cs_scale.to(self.units).f     # Sound speed scale in geometric units
        self._h_scale = self._cs_scale * math.sqrt(8)  # Height scale
        self._vsc_scale = self._cs_scale * self._h_scale * alpha # Viscosity scale
        self._fd_scale = fd_scale.to(self.units).f     # Flux density scale
        self._sd_scale = sd_scale.to(self.units).f / self._vsc_scale # Surface density scale
        self._rm_scale = self._sd_scale / self._h_scale # Rest mass density scale
        self._p_scale = self._cs_scale **2 * self._rm_scale # Pressure scale
        self._kes_scale = kappa_es * (bhU.area / bhU.mass).to(self.units).f # Thomson opacity scale

    def keplerian_omega(self, r: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates the Keplerian angular frequency at a given radius.

        The formula used is :math:`\Omega_K = 1 / r^{1.5}` in geometric units.

        Parameters
        ----------
        r : torch.Tensor
            A tensor of radial coordinates in geometric units.

        Returns
        -------
        torch.Tensor
            A tensor representing the Keplerian angular frequency at each radius.
        """
        return r.pow(-1.5)

    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        # T_eff = T_char * (1 - sqrt(r_in/r))^(1/4) * (r/R_s)^(-3/4)
        # In code units, T_char=1, R_s=2
        outp[mask] = (1 - (self.r_in / r_masked).sqrt()).pow(0.25) * (r_masked / 2.0).pow(-0.75)
        return outp

    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        outp[mask] = self._sf_scale * (r_masked / 2.0).pow(-3) * (1 - (self.r_in / r_masked).sqrt())
        return outp
    
    def sound_speed(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        # c_s = c_s0 * (1 - sqrt(r_in/r))^(1/8) * (r/R_s)^(-3/8)
        # In code units, R_s=2, c_s0 is _cs_scale
        outp[mask] = self._cs_scale * (r_masked / 2.0).pow(-0.375) * (1 - (self.r_in / r_masked).sqrt()).pow(0.125)
        return outp

    def height(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        # h = h_0 * (r/R_s)^(9/8) * (1 - sqrt(r_in/r))^(1/8)
        # In code units, R_s=2, h_0 is _h_scale
        outp[mask] = self._h_scale * (r_masked / 2.0).pow(1.125) * (1 - (self.r_in / r_masked).sqrt()).pow(0.125)
        return outp
    
    def viscosity(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        # nu = nu_0 * (1 - sqrt(r_in/r))^(1/4) * (r/R_s)^(3/4)
        # In code units, R_s=2, nu_0 is _vsc_scale
        outp[mask] = self._vsc_scale * (r_masked / 2.0).pow(0.75) * (1 - (self.r_in / r_masked).sqrt()).pow(0.25)
        return outp
    
    def flux_density(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        outp[mask] = self._fd_scale * (r_masked / 2.0).pow(-4.125) * (1 - (self.r_in / r_masked).sqrt()).pow(0.875)
        return outp

    def surface_density(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        outp[mask] = self._sd_scale * (r_masked / 2.0).pow(-0.75) * (1 - (self.r_in / r_masked).sqrt()).pow(0.75)
        return outp

    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        outp[mask] = self._rm_scale * (r_masked / 2.0).pow(-1.5) * (1 - (self.r_in / r_masked).sqrt()).pow(0.625)
        return outp
    
    def pressure(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        mask = (r >= self.r_in ) & (r <= self.r_cut)
        outp = torch.zeros_like(r)
        r_masked = r[mask]
        outp[mask] = self._p_scale * (r_masked / 2.0).pow(-2.25) * (1 - (self.r_in / r_masked).sqrt()).pow(0.875)
        return outp
    
    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        omega = self.keplerian_omega(r) * self._rot_dir
        g = self.spacetime.g(x)
        g_tt = g[..., 0, 0]
        g_phiphi = g[..., 3, 3]
        
        ut_sq_denom = -(g_tt + omega**2 * g_phiphi)
        # Add protection for numerical instability near horizon where denom -> 0
        ut = torch.sqrt(1.0 / ut_sq_denom.clamp(min=1e-9))
        
        u = torch.zeros_like(x)
        u[..., 0] = ut
        u[..., 3] = omega * ut
        return u

    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x[..., 1] >= self.r_in) & (x[..., 1] <= self.r_cut)
        outp = torch.zeros_like(x[..., 0])
        outp[mask] = self._kes_scale
        return outp