import torch

from bhtrace.geometry.particle._base import Particle, PARTICLE_REGISTRY
from bhtrace.geometry.spacetime._base import Spacetime

from bhtrace.utils.diff import jacobian


@PARTICLE_REGISTRY.register('photon')
class Photon(Particle):
    """Represents a photon, a massless particle.

    This class implements the dynamics of a photon, defined by a Hamiltonian
    where the mass `mu` is zero.

    Attributes
    ----------
    mu : float
        The mass of the particle, always 0 for a photon.
    h : None
        Helicity, not currently implemented.
    """

    def __init__(self, spacetime: Spacetime, **kwargs):
        """Initializes the Photon instance.

        Parameters
        ----------
        spacetime : Spacetime
            The spacetime in which the photon exists.
        **kwargs
            Additional keyword arguments passed to the base class.
        """
        super().__init__(spacetime=spacetime, **kwargs)
        self.mu = 0
        self.h = None

    def hmlt(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculates the Hamiltonian for a photon.

        For a massless photon, the Hamiltonian is `H = 0.5 * g^uv * P_u * P_v`,
        which evaluates to zero for a valid trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates `x^q`, shape [..., 4].
        p : torch.Tensor
            Covariant 4-momentum `p_u`, shape [..., 4].

        Returns
        -------
        torch.Tensor
            The Hamiltonian value, shape [...].
        """
        ginv = self.spacetime.ginv(x)
        return 0.5 * torch.einsum("...uv, ...u, ...v -> ...", ginv, p, p)

    def null_momentum(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Get covariant momentum for a null geodesic (photon).

        Parameters
        ----------
        x : torch.Tensor
            Position.
        v : torch.Tensor
            Direction (contravariant, 4-dimensional, but only spatial part will be used).

        Returns
        -------
        torch.Tensor
            Covariant momentum `p_u`.
        """
        g = self.spacetime.g(x)
        v_spatial = v[..., 1:]

        # For a null vector, g_uv V^u V^v = 0.
        # This is a quadratic equation for V^0:
        # g_00 (V^0)^2 + 2 g_0i V^0 V^i + g_ij V^i V^j = 0

        g00 = g[..., 0, 0]
        g0i = g[..., 0, 1:]
        gij = g[..., 1:, 1:]

        a = g00
        b = 2 * torch.einsum("...i,...i->...", g0i, v_spatial)
        c = torch.einsum("...ij,...i,...j->...", gij, v_spatial, v_spatial)

        # Solve the quadratic equation for V0
        # We choose the positive root for future-pointing vectors.
        v_0 = (-b + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)

        if v_spatial.ndim == 1 and v_0.ndim == 1 and v_spatial.shape[0] != v_0.shape[0]:
            v_spatial = v_spatial.expand(v_0.shape[0], 3)

        v_4 = torch.cat([v_0.unsqueeze(-1), v_spatial], dim=-1)

        p = torch.einsum("...wu, ...u -> ...w", g, v_4)

        return p

    def energy(self, x: torch.Tensor, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Calculates `P_u * u^u`, related to the photon's energy.

        Parameters
        ----------
        x : torch.Tensor
            Spacetime coordinates. Not used.
        p : torch.Tensor
            Covariant 4-momentum `P_u`.
        u : torch.Tensor
            Observer's contravariant 4-velocity `u^u`.

        Returns
        -------
        torch.Tensor
            The scalar product `P_u * u^u`.
        """

        return torch.einsum("...a, ...a -> ...", p, u)

    def spatial_velocity(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculates the spatial velocity of the photon.

        Parameters
        ----------
        X : torch.Tensor
            Position.
        P : torch.Tensor
            Covariant 4-momentum `p_u`.

        Returns
        -------
        torch.Tensor
            The spatial part of the 4-velocity `v^i`.
        """
        ginvX = self.spacetime.ginv(x)
        v = torch.einsum("...uv, ...u -> v", ginvX, p)
        return v[1:]

    def momentum_norm(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Normalizes the spatial part of the momentum.

        Parameters
        ----------
        X : torch.Tensor
            Position.
        P : torch.Tensor
            Covariant 4-momentum `p_u`. The spatial part is modified in-place.

        Returns
        -------
        torch.Tensor
            The momentum with its spatial part normalized.
        """

        ginvX_spatial = self.spacetime.ginv(x)[..., 1:, 1:]
        p2_spatial = torch.einsum(
            "...ij, ...i, ...j ->  ...", ginvX_spatial, p[..., 1:], p[..., 1:]
        )

        p[..., 1:] = p[..., 1:] * torch.pow(p2_spatial, -0.5)

        return p

