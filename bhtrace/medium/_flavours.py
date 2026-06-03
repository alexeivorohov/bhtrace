"""
This module describes main subtypes of Medium class
"""
from abc import abstractmethod 
from typing import Dict, List, Tuple, Optional, Any

import torch

from bhtrace.medium._base import Medium, Spacetime

class ThinDisk(Medium):
    r"""
    Base class for all geometrically thin (H(r) << r) disk models.

    This class provides a foundational structure for modeling thin accretion disks
    around compact objects. 
    Attributes
    ----------
    r_isco : float
        Innermost stable circular orbit (ISCO) radius. This is inferred from
        the spacetime metric.
    r_cut : float
        The outer cutoff radius of the disk. Trajectories outside this radius
        are considered to be outside the disk.
    horizon_tol : float
        Tolerance value used when checking if a point is inside the event horizon.
    _rot_dir : float
        Internal factor indicating the rotation direction of the disk.
        +1.0 for counter-clockwise, -1.0 for clockwise.

    Methods
    -------
    rest_mass_density(x)
        Abstract method: Calculates the rest mass density.
    surface_density(x)
        Abstract method: Calculates the surface density.
    height(x)
        Abstract method: Calculates the vertical height of the disk.
    velocity(x)
        Abstract method: Calculates the 4-velocity of the medium.
    flux_density(x)
        Abstract method: Calculates the radiation flux density.
    pressure(x)
        Abstract method: Calculates the pressure within the disk.
    temperature(x)
        Abstract method: Calculates the temperature of the disk.
    sound_speed(x)
        Abstract method: Calculates the sound speed within the disk.
    metric(x)
        Calculates the spacetime metric at a given point.
    opacity(x)
        Abstract method: Calculates the opacity of the disk.
    hit_condition(s0, s1)
        Determines if a trajectory segment intersects the disk.
    signed_distance(x)
        Calculates the signed coordinate distance to the disk surface.

    Notes
    -----
    This base class assumes the use of Boyer-Lindquist coordinates for
    spacetime points.

    """

    def __init__(
        self,
        spacetime: Spacetime,
        r_cut: Optional[float] = None,
        horizon_tol: float = 1e-2,
        clockwise: bool = False,
        **kwargs
    ):
        """
        Initializes the ThinDisk base class.

        Parameters
        ----------
        spacetime : bhtrace.geometry.Spacetime
            The spacetime in which the thin disk exists.
        r_cut : float, optional
            The outer cutoff radius of the disk. If None, it defaults to
            5 times the ISCO radius.
        horizon_tol : float, optional
            A tolerance value used to determine if a point is effectively
            inside the event horizon, preventing calculations too close to it.
            Defaults to 1e-2.
        clockwise : bool, optional
            If True, the disk is assumed to rotate clockwise. Otherwise,
            counter-clockwise. Defaults to False.
        **kwargs
            Additional keyword arguments passed to the `Medium` base class
            constructor (e.g., `mass`, `temperature`).

        Attributes
        ----------
        r_isco : float
            The innermost stable circular orbit (ISCO) radius, calculated
            from the provided `spacetime` object.
        r_cut : float
            The effective outer cutoff radius of the disk.
        horizon_tol : float
            The tolerance for the event horizon.
        _rot_dir : float
            Internal flag for rotation direction (+1.0 for counter-clockwise,
            -1.0 for clockwise).
        """
        super().__init__(spacetime=spacetime, **kwargs)
        self.r_isco = spacetime.r_isco() # will not hold for 
        self.r_cut = r_cut or 5*self.r_isco
        self.horizon_tol = horizon_tol
        if clockwise:
            self._rot_dir = -1.0
        else:
            self._rot_dir = 1.0
 
  
    @abstractmethod
    def height(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Abstract method: Calculates the vertical height of the disk (H(r)) at
        given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the half-height of the disk
            at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    
    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        
        return torch.sign(s0) != torch.sign(s1)


    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determines the signed coordinate distance to the disk's midplane from
        a given point.

        This is a coordinate distance, not a geodesic distance.
        The input `x` is assumed to be in Boyer-Lindquist spherical
        coordinates (t, r, theta, phi). The signed distance `z` is derived
        from `r*cos(theta)`. Points inside the event horizon (determined by
        `self.spacetime.r_h` and `self.horizon_tol`) are assigned an infinite
        distance to effectively exclude them from disk interaction.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates in spacetime, expected to be in spherical
            Boyer-Lindquist coordinates [t, r, theta, phi], with shape
            [..., 4].
        
        Returns
        -------
        torch.Tensor
            Signed distance of shape [...]. Positive `z` typically means
            above the midplane, negative below. `torch.inf` if inside
            the event horizon.
        """
        r = x[..., 1]
        z = r*torch.cos(x[..., 2])
        z[r < self.spacetime.r_h + self.horizon_tol] = torch.inf

        return z
