from typing import Tuple
import torch
import numpy as np

from bhtrace.medium._base import Medium
from bhtrace.geometry import Spacetime


class PolishDoughnut(Medium):
    """
    A model of optically thick torus

    References
    ----------
        1980, Paczynski & Wiita
    """
    ...