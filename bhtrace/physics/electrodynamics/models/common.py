import torch
import math

from bhtrace.physics.electrodynamics.models._base import Electrodynamics, bhU
from bhtrace.physics.electrodynamics.models.classic import Maxwell


class ModMax(Maxwell):
    """
    Generalized model of ModMax electrodynamics
    
    References
    ----------
    [1] DOI: 10.1103/PhysRevD.102.121703
    
    """

    def __init__(self, units: bhU.UnitSystem, gma = None):
        super().__init__(units)
        self.gma = gma or math.acosh(4 * self.a)
        self.w = math.cosh(gma) * 0.25
        self.h = math.sinh(gma) * 0.25

    def _D(self, F: torch.Tensor, G: torch.Tensor, n: float) -> torch.Tensor:
        return (F.pow(2) + 0.25 * G.pow(2)).pow(n)

    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return - self.w * F + self.h * self._D(F, G, 0.5)

    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return - self.w + self.h * F * self._D(F, G, -0.5)
    
    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return 0.25 * self.h * G * self._D(F, G, -0.5)

    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.h * self._D(F, G, -0.5) - self.h * F.pow(2) * self._D(F, G, -1.5)

    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return - 0.25 * self.h * G * F * self._D(F, G, -1.5)

    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return - 0.0675 * self.h * G.pow(2) * self._D(F, G, -1.5)



class Bardeen(Electrodynamics):
    """
    Bardeen NED model

    First proposed as a source of regular black holes

    References
    ----------
    """

    def __init__(self, g=0, m=1):

        self.g = g
        self.g2 = g**2
        self.s = g / m

        super().__init__()
        if self.g != 0:
            self.l1 = 3 / (2 * self.s * self.g2)
        else:
            self.l1 = 0

    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:

        x = torch.pow(2 * self.g2 * F, -0.5)

        return self.l1 * torch.pow(1 + x, -2.5)

    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:

        x = torch.pow(2 * self.g2 * F, -0.5)

        return self.l1 * 1.25 * torch.pow(1 + x, -3.5) * torch.pow(x, 3)

    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:

        x = torch.pow(2 * self.g2 * F, -0.5)

        term2 = -1.5 * (1 + x) * torch.pow(x, 2)

        return (
            self.l1 * 1.25 * torch.pow(1 + x, -4.5) * torch.pow(x, 3) * (1.75 + term2)
        )

    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)



