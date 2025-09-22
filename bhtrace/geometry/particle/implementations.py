import torch
from .base import Particle
from ..spacetime.base import Spacetime
from bhtrace.functional.diff import Grad


class Photon(Particle):
    """Represents a photon, a massless particle.

    This class implements the dynamics of a photon, defined by a Hamiltonian
    where the mass `mu` is zero.

    Attributes:
        mu (int): The mass of the particle, always 0 for a photon.
        h (None): Helicity, not currently implemented.
    """

    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the Photon instance.

        Args:
            spacetime (Spacetime): The spacetime in which the photon exists.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(spacetime=spacetime, **kwargs)
        self.mu = 0
        self.h = None
  
    def Hmlt(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Calculates the Hamiltonian for a photon.

        For a massless photon, the Hamiltonian is `H = 0.5 * g^uv * P_u * P_v`,
        which evaluates to zero for a valid trajectory.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].

        Returns:
            torch.Tensor: The Hamiltonian value, shape [...].
        """
        ginv = self.spacetime.ginv(X)
        return 0.5 * torch.einsum('...uv, ...u, ...v -> ...', ginv, P, P)

    def dHmlt(self, X: torch.Tensor, P: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Numerically calculates the partial derivatives of the Photon's Hamiltonian.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].
            P (torch.Tensor): Covariant 4-momentum `P_u`, shape [..., 4].
            eps (float, optional): The step size for the finite difference.
                                   Defaults to 1e-4.

        Returns:
            torch.Tensor: The Hamiltonian derivatives `dH/dX^p` at each point,
                          shape [..., 4].
        """

        # Use 2nd order central difference gradient from functional.diff
        def hmlt_func(X):
            return self.Hmlt(X, P)
        grad_calculator = Grad(hmlt_func, eps=eps, order=2)
        
        return grad_calculator(X)

    def GetNullMomentum(self,
                        X: torch.Tensor,
                        V: torch.Tensor
                        ):
        '''
        Get covariant momentum

        Args:
            X: torch.Tensor - position
            V: torch.Tensor - direction (contravariant, 4-dimensional, but only spatial part will be used)
        Returns:
            p: torch.Tensor - covariant momentum (downstairs, contravariant)
        '''
        g = self.spacetime.g(X)
        v_spatial = V[..., 1:]
        
        # For a null vector, g_uv V^u V^v = 0.
        # This is a quadratic equation for V^0:
        # g_00 (V^0)^2 + 2 g_0i V^0 V^i + g_ij V^i V^j = 0
        
        g00 = g[..., 0, 0]
        g0i = g[..., 0, 1:]
        gij = g[..., 1:, 1:]

        a = g00
        b = 2 * torch.einsum('...i,...i->...', g0i, v_spatial)
        c = torch.einsum('...ij,...i,...j->...', gij, v_spatial, v_spatial)

        # Solve the quadratic equation for V0
        # We choose the positive root for future-pointing vectors.
        V0 = (-b + torch.sqrt(b**2 - 4*a*c)) / (2*a)

        if v_spatial.ndim == 1 and V0.ndim == 1 and v_spatial.shape[0] != V0.shape[0]:
            v_spatial = v_spatial.expand(V0.shape[0], 3)

        V4 = torch.cat([V0.unsqueeze(-1), v_spatial], dim=-1)

        p = torch.einsum('...wu, ...u -> ...w', g, V4)
    
        return p

    def energy(self, X, P, u):
        
        return torch.einsum('...v, ...v -> ...', P, u)

    def GetDirection(self, X, P):
        '''
        Inputs:
            X: torch.Tensor - position
            P: torch.Tensor - impulse (downstairs, contravariant)
        '''
        ginvX = self.spacetime.ginv(X)
        v = torch.einsum('...uv, ...u -> v', ginvX, P)
        return v[1:]

    def MomentumNorm(self, X, P):

        ginvX_spatial = self.spacetime.ginv(X)[..., 1:, 1:]
        p2_spatial = torch.einsum('...ij, ...i, ...j ->  ...',
                                  ginvX_spatial, P[..., 1:], P[..., 1:])
        
        P[..., 1:] = P[..., 1:] * torch.pow(p2_spatial, -0.5)
        
        return P     


    def normp(self, X, P):

        pass


class EffPhoton(Particle):

    def __init__(self, spacetime: Spacetime, **kwargs):
        '''
        ### Inputs:
        - spacetime: Spacetime() - spacetime
        '''
        super().__init__(spacetime=spacetime, **kwargs)
        self.mu = 0
        self.h = 0 # helicity
        pass


class PhotonR(Photon):

    def __init__(self, spacetime, **kwargs):

        super().__init__(spacetime=spacetime, **kwargs)
        self.mu = 0
