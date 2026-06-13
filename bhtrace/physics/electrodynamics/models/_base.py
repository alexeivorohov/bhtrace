from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import cached_property

import torch

from bhtrace.geometry.spacetime._base import Spacetime, SpacetimeLocal
from bhtrace.utils import levi_civita_tensor, ClassRegistry
import bhtrace.utils.units as bhU

class Electrodynamics(ABC):
    """Represents certain model of electrodynamics (e.g. Maxwell, Euler-Heisenberg)

    The model of electrodynamics is (completely?) defined by how Lagrangian and it's
    derivatives wrt invariants up to second order are taken.
    """

    def __init__(self, units: bhU.UnitSystem):
        self.units = units

    @abstractmethod
    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor: ...


ELECTRODYNAMICS_REGISTRY = ClassRegistry(Electrodynamics)