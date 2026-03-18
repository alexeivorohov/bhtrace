"""
This module provides a factory function to create particle objects and
exposes the concrete particle classes.
"""

from ._base import Particle, MockParticle, PARTICLE_REGISTRY
from .photon import Photon
