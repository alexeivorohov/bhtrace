'''
Submodule for working with spacetime geometry and fields


'''

from .spacetime import *
from .spacetimes_cart import *
from .spacetimes_sph import *
from .spacetimes_eff import *

from .particle import *
from .particle_zoo import *

from .observer import *
from .spacetime_factory import create_spacetime, SPACETIME_REGISTRY as _SPACETIMES_

_PARTICLES_ = {
    'mock' : None,
    'Photon' : Photon,
}

from .electrodynamics import *
from .electrodynamics_models import *

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


## Status:
# TODO:
# [x] Spacetime baseclass
# [x] Spacetimes base collection
# [X] Baseclass unittests
# [X] Collection unittests
# [x] Descriptions
# [x] Particle baseclass
# [] Particle base collection
# [] Particle baseclass unittests
# [] Collection unittests
# [] Particle descriptions
# [x] Effective geometry baseclass
# [] Effective geometry unittests
# [ ] Models unittests
# [ ] Descriptions