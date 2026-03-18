'''
This file contains Medium class, which describes astrophysical objects

'''

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch

from bhtrace.geometry import Spacetime
from bhtrace.utils import Cacher, Registry


class Medium(ABC):
    """
    Abstract base class for a radiating medium in a spacetime.

    Attributes
    ----------
    cacher : bhtrace.utils.cache.Cacher
        Cache controller

    """

    cacher: Cacher = None

    def __init__(self, spacetime: Spacetime):
        """
        Initializes the medium.

        Parameters
        ----------
        spacetime : Spacetime
            The spacetime in which the medium exists.
        """
        self.spacetime = spacetime

    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the rest mass density of the medium at given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the density at each point.
        """
        raise NotImplementedError

    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def velocity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the 4-velocity of the medium at given spacetime points.

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [..., 4] representing the 4-velocity at each point.
        """
        raise NotImplementedError

    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the radiation flux from the medium at given spacetime points.

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [..., 1] representing the radiation flux at each point.
        """
        raise NotImplementedError
    
    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Determines where a batch of trajectories intersects the medium.

        Parameters
        ----------
        s0 : torch.Tensor
            A tensor of shape [num_steps, num_rays, 4] representing the photon trajectories.

        Returns
        -------
        torch.Tensor(bool)
            A bool tensor mask, where True ind
        """
        raise NotImplementedError
    
    def adjust_hit(
            self, 
            x0: torch.Tensor, 
            x1: torch.Tensor,
            p0: torch.Tensor,
            p1: torch.Tensor,
            s0: torch.Tensor, 
            s1: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines better hit coordinate between two points, using signed distances as weights, then 
        returns best coordinate and interpolated impulse

        Parameters
        ----------
        x0 : torch.Tensor
        x1 : torch.Tensor
        p0 : torch.Tensor
        p1 : torch.Tensor
        s0 : torch.Tensor
        s1 : torch.Tensor

        Notes
        -----
        By default no adjustment is done, and `x1`, `p1` assumed to be hit coordinate.
        This behaviour is natural for volumetric disks.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the adjusted position and momentum.
        """
        return x1, p1
    
    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Detrmines signed coordinate distance to the surface of the object from the given point.

        Notice that this is not geodesic distance - it does not measure real path from the given point to the disk.

        For quasi-flat objects (like thin disks), positive in upper plane, negative on bottom, torch.nan outside the cutoff radius.
        For volumetric objects (like thick disks and tori), negative if the point is inside the object.
        
        Parameters
        ----------
        x : torch.Tensor [..., 4]
            Coordinates in spacetime.
        
        Returns
        -------
        torch.Tensor [...] 
            Signed distance.
        
        '''
        raise NotImplementedError
    
    def __repr__(self):
        ...


MEDIUM_REGISTRY = Registry(Medium)


class Composite(Medium):
    """
    A composition of multiple mediums.
    """

    def __init__(self, spacetime: Spacetime, mediums: Tuple[Medium]):
        raise NotImplementedError
        # super().__init__(spacetime=spacetime)
        # self.mediums = mediums

    def hit(self, x: torch.Tensor):
        """ """
        pass
