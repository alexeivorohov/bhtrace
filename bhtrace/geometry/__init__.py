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


_SPACETIMES_ = {
    'mock': MockSpacetime,
    'MinkowskiCart': MinkowskiCart,
    # 'KerrSchild': KerrSchild,
    # 'SchwSchild': SchwSchild,
    'MinkowskiSph': MinkowskiSph,
    'SphericallySymmetric': SphericallySymmetric,
}

_PARTICLES_ = {
    'mock' : None,
    'Photon' : Photon,
}

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


if __name__ == "__main__":
    
    # Test calls here

    pass