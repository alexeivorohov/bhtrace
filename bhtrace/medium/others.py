from typing import Tuple
import torch
import numpy as np

from bhtrace.medium._base import Medium
from bhtrace.geometry import Spacetime


class SphericalAccretion(Medium):

    def __init__(self, spacetime: Spacetime):
        super().__init__(spacetime)

    def density(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def u_matter(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def flux(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hit(self, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError