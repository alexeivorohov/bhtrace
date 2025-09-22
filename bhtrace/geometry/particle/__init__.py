'''
This module provides a factory function to create particle objects and
exposes the concrete particle classes.
'''

from .base import Particle, MockParticle
from .implementations import Photon, EffPhoton, PhotonR

PARTICLE_REGISTRY = {
    'MockParticle': MockParticle,
    'Photon': Photon,
    'EffPhoton': EffPhoton,
    'PhotonR': PhotonR,
}

def create(name: str, **kwargs):
    '''
    Factory function to create a particle object by name.

    Parameters:
    - name: str - The name of the Particle class to instantiate.
    - **kwargs: Additional keyword arguments to pass to the particle's constructor.

    Returns:
    - An instance of the specified Particle subclass.

    Raises:
    - ValueError: If the specified particle name is not found in the registry.
    '''
    if name not in PARTICLE_REGISTRY:
        raise ValueError(f"Particle '{name}' not recognized. Available particles are: {list(PARTICLE_REGISTRY.keys())}")

    particle_class = PARTICLE_REGISTRY[name]
    return particle_class(**kwargs)
