"""
PyTorch-based library for modeling images of compact objects 
"""

# --- Public API ---

# Submodules

from . import geometry
from . import graphics
from . import grrt
from . import medium
from . import scenarios
from . import tracing
from . import data
from . import utils
from . import globs
from . import registries


# Data structures

from .data import Trajectory, RunningTensor, GRRTData

# Base classes
from .geometry import Spacetime, Particle
from .medium import Medium
from .grrt import RadiativeModel
from .tracing import Tracer

