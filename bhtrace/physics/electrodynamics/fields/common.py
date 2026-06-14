import torch

from bhtrace.physics.electrodynamics.fields._base import ElectromagneticField, Electrodynamics


class PointCharge(ElectromagneticField):

    def __init__(self, q: float = 1.0, g: float = 0.0):
        super().__init__()
        self.q = q
        self.g = g

    def E(self, x: torch.Tensor) -> torch.Tensor:
        outp = torch.zeros_like(x)
        outp[..., 1] = self.q / x[..., 1]
        return outp

    def B(self, x: torch.Tensor) -> torch.Tensor:
        outp = torch.zeros_like(x)
        outp[..., 1] = self.g / x[..., 1]
        return outp


class SplitMonopole(ElectromagneticField):

    def __init__(self, B0: float = 1.0, E0: float = 0.0):
        super().__init__()
        self.B0 = B0
        self.E0 = E0

    def E(self, x: torch.Tensor) -> torch.Tensor:
        outp = torch.zeros_like(x)
        outp[..., 1] = self.B0 / x[..., 1]
        return outp


    def B(self, x: torch.Tensor) -> torch.Tensor:
        outp = torch.zeros_like(x)
        outp[..., 1] = self.E0 / x[..., 1]
        return outp