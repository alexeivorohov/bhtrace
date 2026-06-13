"""
This submodule defines Electrodynamics base class and
implementations of classic and common NED models.

"""

from ._base import Electrodynamics, ELECTRODYNAMICS_REGISTRY

from .classic import (
    Maxwell,
    ParametricPostMaxwell,
    EulerHeisenberg,
    BornInfeld
)

from .common import (
    Bardeen,
    ModMax,
)