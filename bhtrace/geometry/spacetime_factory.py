'''
Provides a factory function to create spacetime objects.
'''

from .spacetime import MockSpacetime
from .spacetimes_cart import MinkowskiCart, KerrSchild, SchwSchild
from .spacetimes_sph import MinkowskiSph, SphericallySymmetric
from .spacetimes_ax import KerrAx
from .spacetimes_eff import EffGeom, EffgeomSimple

SPACETIME_REGISTRY = {
    'MockSpacetime': MockSpacetime,
    'MinkowskiCart': MinkowskiCart,
    'KerrSchild': KerrSchild,
    'SchwSchild': SchwSchild,
    'MinkowskiSph': MinkowskiSph,
    'SphericallySymmetric': SphericallySymmetric,
    # 'KerrAx': KerrAx,
    'EffGeom': EffGeom,
    'EffgeomSimple': EffgeomSimple,
}

def create_spacetime(name: str, *args, **kwargs):
    '''
    Factory function to create a spacetime object by name.

    Parameters:
    - name: str - The name of the spacetime class to instantiate.
    - **kwargs: Additional keyword arguments to pass to the spacetime's constructor.

    Returns:
    - An instance of the specified Spacetime subclass.

    Raises:
    - ValueError: If the specified spacetime name is not found in the registry.
    '''
    if name not in SPACETIME_REGISTRY:
        raise ValueError(f"Spacetime '{name}' not recognized. Available spacetimes are: {list(SPACETIME_REGISTRY.keys())}")

    spacetime_class = SPACETIME_REGISTRY[name]
    return spacetime_class(*args, **kwargs)