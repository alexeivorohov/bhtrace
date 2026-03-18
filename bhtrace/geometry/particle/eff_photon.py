import torch

from bhtrace.geometry.particle._base import Particle, PARTICLE_REGISTRY
from bhtrace.geometry.spacetime._base import Spacetime

from bhtrace.utils.diff import jacobian

@PARTICLE_REGISTRY.register('eff_photon')
class EffPhoton(Particle):
    """Represents a photon in an effective spacetime."""

    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the EffPhoton instance.

        Parameters
        ----------
        spacetime : Spacetime
            The effective spacetime in which the photon exists.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(spacetime=spacetime, **kwargs)
        self.mu = 0
        self.h = 0  # helicity