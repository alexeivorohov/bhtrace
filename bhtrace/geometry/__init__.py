'''
Submodule for working with geometry, particles and fields

By design, this module tries to follow classical field theory, in which particles and fields are constructs over spacetime.


'''

from .spacetime import *
from .spacetimes_cart import *
from .spacetimes_sph import *
from .spacetimes_eff import *

from .particle import *
from .particle_zoo import *

from .coordinates import *
from .coord_systems import *

_SPACETIME_COLLECTION_ = {
    'mock': mock_spacetime,
    'MinkowskiCart': MinkowskiCart,
    # 'KerrSchild': KerrSchild,
    # 'SchwSchild': SchwSchild,
    'MinkowskiSph': MinkowskiSph,
    'SphericallySymmetric': SphericallySymmetric,
}

_COORDS_COLLECTION_ = {
    'mock' : None,
    'Cartesian' : Cartesian
}

_PARTICLE_COLLECTION_ = {
    'mock' : None,
    'Photon' : Photon,

}

## Status:
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
# [] Effective geometry base models
# [] Baseclass unittests
# [] Models unittests
# [] Descriptions
# [] Coordinates base class
# [] Coordinate systems implementations
# [] Coordinate systems unittests
# [] Descriptions


if __name__ == "__main__":
    
    # Test calls here

    pass