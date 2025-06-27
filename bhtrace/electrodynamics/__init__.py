'''
Module for handling different electrodynamics models in a given spacetime
'''
from .electrodynamics import *
from .ed_models import *

_ED_MODELS_ = {
    'Maxwell': Maxwell,
    'ParametricPostMaxwell': ParametricPostMaxwell,
    'EulerHeisenberg': EulerHeisenberg,
    'Bardeen': Bardeen,
    'ModMax': ModMax
    }

### Status:
# [x] One-parameter models
# [] Two-parameter models
# [x] Documentation
# [] Unittests
