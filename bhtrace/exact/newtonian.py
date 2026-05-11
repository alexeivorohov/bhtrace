from typing import Tuple, Optional, Literal

import torch
import torch.linalg as LA


class KeplerianTrajectories:
    """
    Class for handling keplerian trajectories

    Attributes
    ----------

    """

    default_coords = "cart3d"

    def __init__(
        self,
        p: torch.Tensor,
        eps: torch.Tensor,
        nu_0: torch.Tensor = None,
        nu_pe: torch.Tensor = None,
        direction: torch.Tensor = None,
        centers: Optional[torch.Tensor] = None,
        plane_normals: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
    ):
        self.eps = eps
        self.p = p
        _1_m_eps2 = 1 - eps**2
        self.a = p / _1_m_eps2
        self.b = self.p / _1_m_eps2.sqrt()

        if nu_0 is None:
            nu_0 = torch.zeros_like(p)
        self.nu_0 = nu_0

        if nu_pe is None:
            nu_pe = torch.zeros_like(p)
        self.nu_pe = nu_pe

        if direction is None:
            direction = torch.ones_like(p)
        self.direction = direction

        if centers is None:
            centers = torch.zeros([*eps.shape, 3], dtype=eps.dtype, device=eps.device)
        self.centers = centers

        if plane_normals is None:
            plane_normals = self.default_plane_normals(eps.shape, eps.dtype, eps.device)
        self.plane_normals = plane_normals

        self.masses = masses
   
    @classmethod
    def default_plane_normals(cls, shape, dtype, device):
        print(shape, dtype, device)
        plane_normals = torch.zeros([*shape, 3], dtype=dtype, device=device)
        plane_normals[..., 2] = 1.0
        return plane_normals

    @classmethod
    def from_kinematic(
        cls,
        r: torch.Tensor,
        nu: torch.Tensor,
        v_r: torch.Tensor,
        v_nu: torch.Tensor,
        mu: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        plane_normals: Optional[torch.Tensor] = None,
    ) -> "KeplerianTrajectories":
        """
        Initializes KeplerianTrajectories object
        from kinematic parametets.

        Parameters
        ----------
        r : torch.Tensor
            Radius from focus (gravitational center)
        nu : torch.Tensor
            True anomaly
        v_r : torch.Tensor
            Radial speed (derivative of `r`)
        v_nu : torch.Tensor
            Angular speed (derivative of `nu`)
        centers : torch.Tensor, optional (setted to zero)
            2d or 3d batched coordinates of trajectory main focus position.
        plane_normals : torch.Tensor, optional (setted to unit vector in z direction)
            Normal vectors to trajectory's plane.

        Note
        ----
        Since motion of a body in a gravitational field is independent
        of it's mass, all calculations in this step use `m=1`.
        This does not affect units of the evaluated orbit parameters.
        """

        momentum = v_nu * r.pow(2)

        energy = 0.5 * v_r.pow(2) + 0.5 * momentum.pow(2) * r.pow(-2) - mu / r

        print(f"Evaluated energy {energy}")
        print(f"Evaluated momentum {momentum}")
        tol = 1e-3
        spherical_mask = torch.less(abs(energy), tol)
        ellipsis_mask = torch.less(energy, -tol)
        hyperbola_mask = torch.greater(energy, tol)
        direction = torch.sign(v_nu)

        nu_pe = torch.zeros_like(r)
        eps = torch.zeros_like(r)
        # TODO: add correct handling of initialization from periapsis and apoapsis
        # (now it counts r_dt = 0  as circle)
        p = momentum.pow(2) / mu
        eps = (1 + 2 * energy * momentum.pow(2) / mu.pow(2)).sqrt()

        if torch.any(ellipsis_mask):
            th = cls._ellipsis_true_anomaly(
                r[ellipsis_mask],
                v_r[ellipsis_mask],
                v_nu[ellipsis_mask],
                p[ellipsis_mask],
                eps[ellipsis_mask],
            )
            # theta = nu + nu0
            nu_pe[ellipsis_mask] = th - nu[ellipsis_mask]

        print(f"Evaluated p: {p}")
        print(f"Evaluated eps: {eps}")
        print(f"Evaluated nu_0: {nu}")
        print(f"Evaluated nu_pe: {nu_pe}")

        return cls(
            eps=eps,
            p=p,
            nu_0=nu,
            nu_pe=nu_pe,
            direction=direction,
            centers=centers,
            plane_normals=plane_normals,
            masses=mu,
        )

    @classmethod
    def _ellipsis_true_anomaly(
        cls,
        r: torch.Tensor,
        v_r: torch.Tensor,
        v_th: torch.Tensor,
        p: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        cos_th = (p / r - 1) / eps
        sin_th = p * v_r / (eps * r.pow(2) * v_th)
        print(f"cos_th: {cos_th}")
        print(f"sin_th: {sin_th}")

        return torch.atan2(sin_th, cos_th)

    @classmethod
    def from_cartesian(
        cls,
        x: torch.Tensor,
        v: torch.Tensor,
        mu: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        plane_normals: Optional[torch.Tensor] = None,
    ) -> "KeplerianTrajectories":

        if v.shape != x.shape:
            raise ValueError(f"Dimension mismatch for x and v: {x.shape} != {v.shape}")

        if x.shape[-1] == 3:
            if plane_normals is not None:
                raise ValueError(
                    "Can not set `plane_normals` manually for 3d coordinates `x`"
                )
            plane_normals = x.cross(v, dim=-1)
            plane_normals = plane_normals / LA.vector_norm(plane_normals, dim=-1, ord=2)
            # perform projection

        elif x.shape[-1] == 2:
            # x = torch.concat([x, torch.zeros([*x.shape[:-1], 1])], dim=-1).to(dtype=x.dtype, device=x.device)
            # v = torch.concat([v, torch.zeros([*v.shape[:-1], 1])], dim=-1).to(dtype=v.dtype, device=v.device)
            plane_normals = cls.default_plane_normals(x.shape[:-1], x.dtype, x.device)

        return cls.from_cartesian_2d(
            x[..., 0], x[..., 1], v[..., 0], v[..., 1], mu, centers, plane_normals
        )

    @classmethod
    def from_cartesian_2d(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        v_x: torch.Tensor,
        v_y: torch.Tensor,
        mu: torch.Tensor,
        centers: torch.Tensor = None,
        plane_normals: torch.Tensor = None,
    ) -> "KeplerianTrajectories":

        r = (x.pow(2) + y.pow(2)).sqrt()
        sin_nu, cos_nu = y / r, x / r
        nu = torch.atan2(y, x)
        v_r = v_x * cos_nu + v_y * sin_nu
        v_nu = -v_x * sin_nu + v_y * cos_nu
        print(f"Polar coords (r, nu): {r, nu}")
        print(f"Polar velocities (v_r, v_nu): {v_r, v_nu}")
        return cls.from_kinematic(r, nu, v_r, v_nu, mu, centers, plane_normals)

    def sample(
        self,
        t: torch.Tensor,
        nu0: Optional[torch.Tensor] = None,
        coords: Literal["polar", "cart2d", "cart3d"] = None,
    ) -> torch.Tensor: ...

    def propagate_true_anomaly(
        self,
        nu: torch.Tensor,
        nu_0: torch.Tensor = None,
    ) -> torch.Tensor:
        if nu_0 is None:
            nu_0 = self.nu_0
        return nu * self.direction + nu_0

    def phase_sample(
        self,
        nu,
        nu_0: Optional[torch.Tensor] = None,
        coords: Literal["polar", "cart2d", "cart3d"] = None,
    ) -> torch.Tensor:

        nu = self.propagate_true_anomaly(nu, nu_0)

        r = self.p / (1 + self.eps * torch.cos(nu + self.nu_pe))

        return self._projection(r, nu, coords)

    def _projection(
        self,
        r: torch.Tensor,
        nu: torch.Tensor,
        coords: Literal["polar", "cart2d", "cart3d"] = None,
    ) -> torch.Tensor:

        if coords == "polar":
            return torch.stack([r, nu], dim=-1)

        if coords == "cart2d":
            x = r * torch.cos(nu)
            y = r * torch.sin(nu)
            return torch.stack([x, y], dim=-1)

        # if coords == 'cart3d':

        raise ValueError(f"Unsupported coordinates: {coords}")


if __name__ == "__main__":
    ...