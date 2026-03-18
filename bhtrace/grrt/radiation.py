r"""

Notes
-----
Currently, this module relies entirely on the assumption of local thermodynamic
equilibrium, which allows relating emission and absorption coefficients as:

.. math:: \frac{j_{\nu}}{\alpha_{\nu}} = B(\nu, T)

and thus radiative transfer equation can be reduced to

.. math::

    \mathcal{I}_{i+1}(\nu) = \mathcal{I}_{\nu, i}(\nu) e^{-\Delta\tau_i(\nu)} + B(\nu, T) (1 - e^{-\Delta\tau_i(\nu)})
"""

from abc import ABC, abstractmethod
import inspect

from scipy import constants
import torch

from bhtrace.utils import Registry
from bhtrace.geometry.observer import Observer
from bhtrace.geometry.spacetime import Spacetime
from bhtrace.medium import Medium


class RadiativeModel(ABC):
    r"""
    Base class of radiation models to be used with `GRRT` class.

    Peovides an additional level of abstraction for `GRRT` calculations.
    Uses local physical properties of the medium to compute local radiative properties
    according to certain model and then provides high-level api for radiative transfer solver.

    Attributes
    ----------
    spectral : bool 
        Flag indicating if this model calculates radiation spectrum or total flux.

    Methods
    -------

    step


    References
    ----------

    """

    def __init__(self, spectral: bool):
        self.spectral = spectral

    @abstractmethod
    def step(
        self,
        inv_i_prev: torch.Tensor,
        nu_comoving: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        invariant_i_prev : torch.Tensor (..., n_freq)
            Invariant intensity on previous step
        nu_comoving : torch.Tensor (..., n_freq)
            Frequency in comoving (fluid) frame
        dlambda : torch.Tensor (...)
            Affine parameter differential

        Returns
        -------
        torch.Tensor (..., n_freq)
            Invariant intensity

        """
        ...

RADIATIVE_MODEL_REGISTRY = Registry(RadiativeModel)


@RADIATIVE_MODEL_REGISTRY.register("blackbody")
class Blackbody(RadiativeModel):

    _a = 2.0 * torch.pi * constants.h / constants.c**2
    _b = constants.h / constants.Boltzmann

    def __init__(self, dtau_thick=1e-2):
        super().__init__(spectral=True)
        self.dtau_thick = dtau_thick

    def step(
        self,
        x: torch.Tensor,
        inv_i_prev: torch.Tensor,
        nu_comoving: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:

        temp = medium.temperature(x).unsqueeze(-1)
        rest_mass_density = medium.rest_mass_density(x).unsqueeze(-1)
        opacity = medium.opacity(x).unsqueeze(-1)
        inv_alpha = rest_mass_density * opacity * nu_comoving

        mu = self._b * nu_comoving / temp
        inv_source = self._a * nu_comoving.pow(-3) / torch.expm1(mu)

        dtau = inv_alpha * dlambda.unsqueeze(-1)

        new_inv_i = torch.zeros_like(inv_i_prev)

        thick = dtau > self.dtau_thick
        thin = ~ thick

        if torch.any(thin):
            dtau_thin = dtau[thin]
            new_inv_i[thin] = inv_i_prev[thin] + (inv_source[thin] - inv_i_prev[thin]) * dtau_thin

        if torch.any(thick):
            dtau_thick = dtau[thick]
            exp_dtau_thick = (-dtau_thick).exp()
            new_inv_i[thick] = inv_i_prev[thick] * exp_dtau_thick + inv_source[thick] * (1.0 - exp_dtau_thick)

        return new_inv_i

@RADIATIVE_MODEL_REGISTRY.register("bolometric_flux")
class IntegralFlux(RadiativeModel):
    """
    Uses medium's surface flux method to evaluate radiative transfer along the ray. 
    Assumes that the medium is optically thin and thus ignores absorption.

    References
    ----------
    """

    def __init__(self):
        super().__init__(spectral=False)

    def step(
        self,
        x: torch.Tensor,
        inv_i_prev: torch.Tensor,
        z: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:
        
        flux_from_medium = medium.surface_flux(x)
        
        doppler_factor = (1+z).pow(-4)

        d_inv_i = doppler_factor * flux_from_medium
        # Filter out NaNs, replacing them with 0
        d_inv_i = torch.nan_to_num(d_inv_i, nan=0.0)
        
        result = inv_i_prev + d_inv_i
        return result
