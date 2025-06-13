'''
Submodule for working with geometry, particles and fields

By design, this module tries to follow classical field theory, in which particles and fields are constructs over spacetime.


'''

from .spacetime import *
from .collection_cart import *
from .collection_sph import *
from .effgeom import *
from .particle import *
from .particle_zoo import *

ST_COLLECTION = {
    'mock_spacetime': mock_spacetime,
    'MinkowskiCart': MinkowskiCart,
    # 'KerrSchild': KerrSchild,
    # 'SchwSchild': SchwSchild,
    'MinkowskiSph': MinkowskiSph,
    'SphericallySymmetric': SphericallySymmetric,
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
# [] Descriptions
# [x] Effective geometry baseclass
# [] Effective geometry base models
# [] Baseclass unittests
# [] Models unittests
# [] Descriptions


if __name__ == "__main__":
    
    # Test calls here

    pass