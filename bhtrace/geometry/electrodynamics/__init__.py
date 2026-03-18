'''
This module provides a factory function to create electrodynamics model objects and
exposes the concrete model classes.
'''

from ._base import Electrodynamics, ELECTRODYNAMICS_REGISTRY
from .models import (
    Maxwell,
    EulerHeisenberg,
    BornInfeld,
    ModMax,
    Bardeen,
    ParametricPostMaxwell,
)

ELECTRODYNAMICS_REGISTRY.update({
    'Maxwell': Maxwell,
    'EulerHeisenberg': EulerHeisenberg,
    'BornInfeld': BornInfeld,
    'ModMax': ModMax,
    'Bardeen': Bardeen,
    'ParametricPostMaxwell': ParametricPostMaxwell,
})