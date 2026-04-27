"""
This module describes ThinDisk baseclass and basic accretion models, including:
* KeplerianDisk
* AlphaDisk

***For now, all disks should use spherical coordinates.***

Workflow in `bhtrace` allows to completely detach units, for which ray-tracing is done
(geometrized) and units in which GRRT calculations are performed. At this stage, all GRRT
calculations are done in SI.

"""

from typing import Dict, List, Tuple, Optional, Any
import math

import torch
from typing import Optional

from bhtrace.medium._base import Medium, Spacetime, Cacher

class ThinDisk(Medium):
    r"""
    Base class for all geometrically thin (H(r) << r) disk models

    This implementation mostly according to_[1]

    Attributes
    ----------
    r_isco : float
        Innermost stable circulat orbit radius. Inferred from metric.
    r_cut : float
        Cutoff radius of the disc
    m_dot : float
        Dimensionless accretion rate [\dot{m} = \dot{M}c^2/L_{edd}]
    a : float 
        Dimensionless rotation parameter of the metric. Zero if not provided.
    

    Methods
    -------


    Notes
    -----
    Uses Boyer-Lindquist coordinates

    References
    ----------
    ...:
    [1] https://arxiv.org/pdf/1104.5499
    """

    cacher = Cacher(False)

    def __init__(
            self,
            spacetime: Spacetime,
            r_cut: Optional[float] = None,
            m_dot: Optional[float] = None,
        ):
        self.spacetime = spacetime
        self.r_isco = spacetime.r_isco() # will not hold for 
        self.r_cut = r_cut or 5*self.r_isco
        self.m_dot = m_dot or 1e-1 # Default value? Dimensionless accr. rate: 
        self.a = getattr(spacetime, 'a', 0.0)
 
    def rest_mass_density(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def surface_density(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def height(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def flux_density(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def pressure(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def temperature(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return NotImplementedError

    def metric(self, x: torch.Tensor) -> torch.Tensor:
        return self.spacetime.g(x)
    
    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        
        return torch.sign(s0) != torch.sign(s1)

    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        z = r*torch.cos(x[..., 2])
        z[r < self.spacetime.r_h] = torch.inf
        return z
        

class KeplerianDisk(ThinDisk):
    """
    Physically motivated Keplerian thin disk in geometric units (G=c=M=1).
    Schwarzschild approximation; vertically isothermal Gaussian structure.
    """
    def __init__(
        self,
        spacetime,
        r_cut: Optional[float] = None,
        m_dot: float=0.1
    ):
        self.r_cut = r_cut
        self.spacetime = spacetime
        # Normalization constants (geometric units)
        self.r_isco = 6
        self.m_dot = m_dot
        self.f_norm = 3 / (8 * torch.pi) * self.m_dot  # SS flux prefactor
        self.sigma_norm = 5.67e-5  # Stefan-Boltzmann / (proper conversion to geom.)
        # Note: sigma in geom. units ~ 10^{-7}; tune empirically or derive exactly

    def keplerian_omega(self, r: torch.Tensor) -> torch.Tensor:
        """Keplerian angular freq. Omega_K = 1/sqrt(r^3) (Schwarzschild approx.)"""
        return r.pow(-1.5)

    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        r_in = self.r_isco
        r_out = self.r_cut

        mask = (r >= r_in) & (r <= r_out)
        F = torch.zeros_like(r)
        F_loc = self.f_norm * (r.pow(-3) * (1 - (r_in / r).sqrt()))
        F = torch.where(mask, F_loc, torch.zeros_like(F_loc))
        return F

    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Effective temperature from F = sigma T^4."""
        F = self.surface_flux(x)
        return (F / self.sigma_norm).pow(0.25)

    def height(self, x: torch.Tensor) -> torch.Tensor:
        """Scale height H/r ~ c_s / (Omega_K r)."""
        r = x[..., 1]
        T = self.temperature(x)
        cs = (self.kB_mu_mp * T).sqrt()
        Omega = self.keplerian_omega(r)
        return cs / Omega

    def surface_density(self, x: torch.Tensor) -> torch.Tensor:
        """Sigma approx from optically thick tau ~ kappa_es Sigma / 2 ~ 10-100."""
        r = x[..., 1]
        H_over_r = self.height(x)[..., None] / r[..., None]  # broadcast
        # SS scaling Sigma ~ alpha^{-1} m_dot (H/r)^{-2} r^{-3/2} (rough)
        Sigma_norm = self.m_dot / (self.alpha * H_over_r.pow(2))
        return Sigma_norm * r.pow(-1.5)

    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian vertical profile."""
        r = x[..., 1]
        z = r * torch.cos(x[..., 2])
        H = self.height(x)
        Sigma = self.surface_density(x)
        rho0 = Sigma / (torch.sqrt(2 * torch.pi) * H)
        return rho0 * torch.exp(-0.5 * (z / H).pow(2))

    def pressure(self, x: torch.Tensor) -> torch.Tensor:
        """Ideal gas P = rho kT / (mu m_p)."""
        rho = self.rest_mass_density(x)
        T = self.temperature(x)
        return rho * self.kB_mu_mp * T

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        """Keplerian 4-vel. u^mu = (u^t, 0, 0, u^phi); Schwarzschild approx."""
        r = x[..., 1]
        Omega = self.keplerian_omega(r)
        # For Schwarzschild: u^t ~ 1/sqrt(1-3/r), u^phi = Omega u^t (midplane approx)
        # Full: use metric as before
        g = self.metric(x)
        g_tt = g[..., 0, 0]
        g_phiphi = g[..., 3, 3]
        u_t = ( -(g_tt + Omega**2 * g_phiphi) ).sqrt()
        u_phi = Omega * u_t

        u = torch.zeros_like(x)
        u[..., 0] = u_t   # contravariant? Normalize properly if needed
        u[..., 3] = u_phi
        return u

    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        """Dominant Thomson opacity."""
        return torch.full_like(x[..., 0], self.kappa_es)

    def viscosity(self, x: torch.Tensor) -> torch.Tensor:
        """alpha viscosity nu = alpha c_s H."""
        r = x[..., 1]
        T = self.temperature(x)
        cs = (self.kB_mu_mp * T).sqrt()
        H = self.height(x)
        return self.alpha * cs * H

    def flux_density(self, x: torch.Tensor) -> torch.Tensor:
        """Dissipative heating Q_visc ~ (9/4) nu Sigma Omega^2 (vert. integrated)."""
        r = x[..., 1]
        Sigma = self.surface_density(x)
        Omega = self.keplerian_omega(r)
        nu = self.viscosity(x)
        return (9/4) * nu * Sigma * Omega.pow(2)


class AlphaDisk(ThinDisk):
    """
    Shakura-Synaev alpha disk
    
    Attributes
    ----------



    """
    def __init__(
            self, 
            spacetime,
            alpha: float = 0.5,
            m_dot: float = 16.0,

        ):
        super().__init__(
            spacetime=spacetime,

        )
        self.alpha = alpha
        self._a2 = self.a**2
        self._a4 = self._a2**2
        acos_a = math.acos(self.a)
        self._r_ast = ...
        y = [
            self.r_isco ** 0.5, 
            2*math.cos((acos_a - math.pi) / 3),
            2*math.cos((acos_a + math.pi) / 3),  
            -2*math.cos(acos_a / 3),
        ]
        assert all([_y > 1e-7 for _y in y]), f"Low values in y list: {y}"
        self._y = y
        self.kappa = {
            'outer': 0.0, #free-free opacity 
            'middle': 0.0, #electron-scattering opacity 0.34 cm2 g-1
            'inner': 0.0, #electron-scattering opacity 0.34 cm2 g-1
        }
        self._qk_coefs = [
            1.5*self.a, 
            3*(y[1] - self.a)**2/y[1]*(y[1]-y[2])*(y[1]-y[3]),
            3*(y[2] - self.a)**2/y[2]*(y[2]-y[1])*(y[2]-y[3]),
            3*(y[3] - self.a)**2/y[3]*(y[3]-y[1])*(y[3]-y[2]),
        ] # page 29
        assert all([v > 1e-7 for v in self._qk_coefs]), f"Low values in y list: {y}"

        self.f0 = 1.0
        self.sgma0 = 1.0
        self.H0 = 1.0
        self.rho0 = 1.0
        self.T0 = 1.0

    def _region(self, x: torch.Tensor):
        ...

    def flux(self, x: torch.Tensor):
        ...

    def radial_a(self, y: torch.Tensor):
        return 1.0 + self._a2*(y**(-4) + 2 * y**(-6))
    
    def radial_b(self, y: torch.Tensor):
        return 1.0 + self.a * y**(-3)

    def radial_c(self, y: torch.Tensor):
        return 1.0 - 3.0 * y**(-2) + 2 * self._a2 * y**(-2)

    def radial_d(self, y: torch.Tensor):
        return 1.0 - 2 * y**(-2) + self._a2 * y**(-4)
    
    def radial_e(self, y: torch.Tensor):
        return 1.0 + 4*self._a2 * y**(-4) * (1 - y**(-2)) + 3*self._a4 * y**(-8)
    
    def radial_q(self, y: torch.Tensor):
        _y = self.y
        _q = self._qk_coefs
        q0 = 1/y * (1 + self.a * y**(-3)) * torch.pow(1 - 3*y**(-2) + 2*self.a*y**(-2), -0.5)
        mul = y - _y[0] - _q[0]*torch.log(y/_y[0]) - _q[1]*torch.log((y-_y[1])/(y[0]-y[1]))
        mul+= - _q[2](torch.log((y-_y[2])/(y[0]-y[2]))) + _q[3]*torch.log((y-_y[3])/(y[0]-y[3]))

        return q0*mul
