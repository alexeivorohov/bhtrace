'''
Module for handling different electrodynamics models in a given spacetime

Goes with collection of different basic models.

By default:
    mu_0 = 4pi, epsilon_0 = 1/{4pi}, c=1
    E_c = ?


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
