from typing import Tuple, Optional, Literal

import torch
import torch.linalg as LA


class EllipticTrajectories:
    """
    Class for handling elliptic trajectories

    Attributes
    ----------
    
    eps : torch.Tensor
        Eccentricity

    p : torch.Tensor
        Semi-latus

    kappa : torch.Tensor
        Geometric parameter

    """
    default_coords = 'cart3d'

    def __init__(
            self,
            eps: torch.Tensor,
            p: torch.Tensor,
            kappa: torch.Tensor,
            centers: Optional[torch.Tensor] = None,
            plane_normals: Optional[torch.Tensor] = None,
        ):

        self.eps = eps
        self.p = p
        self.kappa = kappa
        
        if centers is None:
            centers = torch.zeros([*eps.shape, 3]).to(dtype=p.dtype, device=p.device)      
        self.centers = centers

        if plane_normals is None:
            plane_normals = self.default_plane_normals(eps.shape, eps.device, eps.dtype)       
        self.plane_normals = plane_normals

    @classmethod
    def default_plane_normals(cls, shape, dtype, device):
        plane_normals = torch.zeros([*shape, 3]).to(dtype=dtype, device=device)
        plane_normals[..., 2] = 1.0
        return plane_normals


    @classmethod
    def from_kinematic(
        cls,
        r: torch.Tensor,
        nu: torch.Tensor,
        v_r: torch.Tensor,
        v_nu: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        plane_normals: Optional[torch.Tensor] = None,
    ) -> 'EllipticTrajectories':
        """
        Initializes EllipticTrajectories2d object
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
        """
        
        kappa = v_nu*r.pow(2)
        
        eps = torch.zeros_like(r)
        # TODO: add correct handling of initialization from periapsis and apoapsis
        # (now it counts r_dt = 0  as circle)
        mask = torch.greater(abs(v_r), 1e-6) 
        
        if torch.any(mask):
            a = kappa[mask]*torch.sin(nu[mask]) / v_r[mask]
            b = r[mask] * torch.cos(nu[mask])
            print(a.shape)
            print(b.shape)
            eps[mask] = r[mask] / (a - b)

        p = r*(1 + eps*torch.cos(nu))

        return cls(eps, p, kappa, centers, plane_normals)

    @classmethod
    def from_cartesian(
        cls,
        x: torch.Tensor,
        v: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        plane_normals: Optional[torch.Tensor] = None,
    ) -> 'EllipticTrajectories':
        
        if v.shape != x.shape:
            raise ValueError(
                f'Dimension mismatch for x and v: {x.shape} != {v.shape}' 
            )

        if x.shape[-1] == 3:
            if plane_normals is not None:
                raise ValueError(
                    'Can not set `plane_normals` manually for 3d coordinates `x`'
                )
            plane_normals = x.cross(v, dim=-1)
            plane_normals = plane_normals / LA.vector_norm(plane_normals, dim=-1, ord=2)

        elif x.shape[-1] == 2:
            x = torch.concat([x, torch.zeros([*x.shape[:-1], 1])], dim=-1).to(dtype=x.dtype, device=x.device)
            v = torch.concat([v, torch.zeros([*v.shape[:-1], 1])], dim=-1).to(dtype=v.dtype, device=v.device)
            plane_normals = cls.default_plane_normals(x.shape[:-1], x.dtype, x.device)

        r = LA.vector_norm(x, dim=-1, ord=2).unsqueeze(-1)
        nu = torch.ones_like(r)*torch.pi/2 # ???

        e_r = x / r
        e_phi = e_r.cross(plane_normals, dim=-1)

        v_r = torch.sum(v * e_r, dim=-1).unsqueeze(-1)
        v_nu = torch.sum(v * e_phi, dim=-1).unsqueeze(-1)

        print(r.shape, nu.shape, v_r.shape, v_nu.shape)

        return cls.from_kinematic(r, nu, v_r, v_nu, centers, plane_normals)


    def sample(
        self,
        t: torch.Tensor,
        nu0: Optional[torch.Tensor] = None,
        coords: Literal['polar', 'cart2d', 'cart3d'] = None,
    ) -> torch.Tensor:
        

        ...
        
    def phase_sample(
        self,
        nu: torch.Tensor = None,
        coords: Literal['polar', 'cart2d', 'cart3d'] = None,
    ) -> torch.Tensor:
        
        r = self.p / (1 + self.eps*torch.cos(nu))

        return self._projection(r, nu, coords)

    def _projection(
            self, 
            r: torch.Tensor, 
            nu: torch.Tensor, 
            coords: Literal['polar', 'cart2d', 'cart3d'] = None
        ) -> torch.Tensor:

        if coords == 'polar':
            return torch.stack([r, nu], dim=-1)
        
        if coords == 'cart2d':
            x = r*torch.cos(nu)
            y = r*torch.sin(nu)
            return torch.stack([x, y], dim=-1)

        # if coords == 'cart3d':


        raise ValueError(
            f'Unsupported coordinates: {coords}'
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 16
    x = torch.stack(
        [20*torch.ones(N), torch.linspace(5, 10, N)], dim=-1
    )

    v = torch.stack(
        [-0.1*torch.ones(N), torch.zeros(N)], dim=-1
    )

    elliptic = EllipticTrajectories.from_cartesian(x, v)

    nu = torch.linspace(0, torch.pi*2, 128)

    traj = elliptic.phase_sample(nu, 'cart2d')

    x_, y_ = traj.unbind(dim=-1)
    
    print(x_.shape, y_.shape)
    plt.plot(x_.T, y_.T)
    plt.show()

