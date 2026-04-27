from typing import Tuple
import torch
import numpy as np

from bhtrace.medium._base import Medium
from bhtrace.geometry import Spacetime



class SlimDisk(Medium):
    """
    Accretion disk with advection accounted

    References
    ----------
    1988, Ambramoviz & Fragile

    """
    ...