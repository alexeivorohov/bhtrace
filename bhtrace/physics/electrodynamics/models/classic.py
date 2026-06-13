import torch

from bhtrace.physics.electrodynamics.models._base import Electrodynamics, bhU


class Maxwell(Electrodynamics):
    """
    Classical Maxwell electrodynamics
    """

    def __init__(self, units: bhU.UnitSystem):

        super().__init__(units)
        self.a = - (0.25 / torch.pi / bhU.eps0.to(self.units)).f

    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.a * F

    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.fill(F, self.a)

    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)


class ParametricPostMaxwell(Electrodynamics):
    """Parametric Post-Maxwell Electrodynamics

    This NED model is accounting for generic second-order (w.r.t. invariants) 
    corrections to the Maxwellian model.
    """

    def __init__(self, units: bhU.UnitSystem, eta_1: float = 1, eta_2: float = 0):

        super().__init__(units)
        self.a = - (0.25 / torch.pi / bhU.eps0.to(self.units)).f
        self.eta_F = eta_1
        self.eta_G = eta_2
        self.eta_2F = self.eta_F * 2
        self.eta_2G = self.eta_G * 2
    
    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.a * F + self.eta_F * F.pow(2) + self.eta_G * G.pow(2) 

    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.a + self.eta_2F * F 

    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.eta_2G * G

    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.fill(F, self.eta_2F)

    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(F)

    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.fill(G, self.eta_2G)


class EulerHeisenberg(ParametricPostMaxwell):
    """
    Effective, low-energy limit of Euler-Heisenberg-Kockel theory.
    
    """

    def __init__(self, units: bhU.UnitSystem, scale: float = 1.0):
        """
        Parameters
        ----------
        units : bhtrace.utils.units.UnitSystem
            Unit system to use for the model.
        scale : float, defaults to 1.0
            Multiplier for non-linear terms in theory Lagrangian.
        """

        h = (bhU.alpha / bhU.mu0 / bhU.schwinger_E).pow(2).to(units)
        eta_1 = h.f * scale / 45
        eta_2 = eta_1 * 7 / 4

        super().__init__(units, eta_1=eta_1, eta_2=eta_2)



class BornInfeld(Electrodynamics):
    """
    Born-Infeld Nonlinear Electrodynamics    
    
    """


    def __init__(self, units: bhU.UnitSystem, scale: float =1.0):
        """
        Parameters
        ----------
        scale : float, defaults to 1.0
            Multiplier for critical field in theory lagrangian
        """
        super().__init__(units)
        self.b = bhU.born_infeld_E.to(units).f * scale
        self.b2 = self.b ** 2
        self.alpha = 1.0 / (2.0 * self.b2)
        self.gamma = - self.alpha ** 2

    def _D(self, F, G):
        return 1.0 + self.alpha * F + self.gamma * (G**2)

    def L(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return self.b2 * (1.0 - torch.sqrt(self._D(F, G)))

    def L_F(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.alpha * self.b2 * torch.pow(self._D(F, G), -0.5)

    def L_G(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        D = self._D(F, G)
        return -self.gamma * self.b2 * (torch.pow(D, -0.5) - self.gamma * (G**2) * torch.pow(D, -1.5))

    def L_FF(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return 0.25 * (self.alpha**2) * self.b2 * torch.pow(self._D(F, G), -1.5)

    def L_FG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.alpha * self.gamma * self.b2 * G * torch.pow(self._D(F, G), -1.5)

    def L_GG(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        D = self._D(F, G)
        return -self.gamma * self.b2 * (torch.pow(D, -0.5) - self.gamma * (G**2) * torch.pow(D, -1.5))
