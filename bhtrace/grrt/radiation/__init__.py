r"""
The `radiation` module provides models for radiative transfer calculations
within the `bhtrace` framework. These models are used by the `GRRT` (General
Relativistic Radiative Transfer) solver to compute the change in intensity of
light rays as they propagate through a medium.

This module is built upon the `RadiativeModel` base class, which defines a
common interface for all radiation models. Each model implements the `step`
method, which calculates the new intensity at a point along a ray's path.

Available Models
----------------
- `BolometricFlux`: A simple model for bolometric flux that assumes an
  optically thin medium. It is useful for scenarios where absorption can be
  neglected.
- `Blackbody`: A model for blackbody radiation that includes both emission and
  absorption. It correctly handles optically thin and thick regimes.
- `BlackbodyN`: A test model for blackbody radiation that only includes
  emission, ignoring absorption.

Notes
-----
Currently, this module relies entirely on the assumption of local thermodynamic
equilibrium (LTE), which allows relating emission and absorption coefficients as:

.. math::
    \frac{j_{u}}{\alpha_{u}} = B(u, T)

where :math:`j_{u}` is the emission coefficient, :math:`\alpha_{u}` is the 
absorption coefficient, and :math:`B(u, T)` is the Planck function.

Under this assumption, the radiative transfer equation can be solved as:

.. math::
    \mathcal{I}_{i+1}(u) = \mathcal{I}_{i}(u) e^{-\Delta \tau_i(u)} + S_i(u) (1 - e^{-\Delta \tau_i(u)})

where :math:`\mathcal{I}_{i}(u)` is the specific intensity at step :math:`i`, :math:`\Delta	\tau_i(u)` 
is the optical depth of the step, and :math:`S_i(u)` is the source function, which in LTE is equal 
to the Planck function :math:`B(u, T)`.
"""

from ._base import RadiativeModel, RADIATIVE_MODEL_REGISTRY
from .bolometric import BolometricFlux
from .blackbody import Blackbody

# Alias for backward compatiblity
IntegralFlux = BolometricFlux
