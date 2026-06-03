"""
This module contains the `Medium` base class, which describes the properties
of astrophysical objects that act as a radiating medium in a spacetime.

The `Medium` class provides an abstract interface for various physical
properties such as rest mass density, temperature, 4-velocity, surface flux,
flux density, and opacity. It also defines methods for determining
intersections with the medium and adjusting hit coordinates.

The module also introduces `GRRTUnitSystem`, a specialized unit system tailored
for General Relativistic Radiative Transfer (GRRT) calculations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch

from bhtrace.geometry import Spacetime
from bhtrace.utils import Registry
import bhtrace.utils.units as bhU

class Medium(ABC):
    R"""
    Abstract base class for a radiating medium in a spacetime.

    This class defines the fundamental properties and behaviors expected of
    any medium that interacts with radiation within a given spacetime.
    Subclasses must implement abstract methods to provide specific physical
    characteristics.

    Attributes
    ----------
    mass : float
        The characteristic mass of the astrophysical object in SI units (kg).
    spacetime : bhtrace.geometry.Spacetime
        The spacetime geometry in which the medium exists.
    units : GRRTUnitSystem
        The unit system tailored for GRRT calculations, derived from the
        `mass` and `temperature` provided during initialization.
    _nu_scale : float
        Frequency scaling factor for transforming from SI to medium physical units.
    _bb_scale : float
        Scaling factor related to the Planck function for blackbody radiation.
    _bb_pow : float
        Power factor related to the Planck function for blackbody radiation.

    Methods
    -------
    rest_mass_density(x)
        Calculates the rest mass density at given spacetime points.
    temperature(x)
        Calculates the temperature at given spacetime points.
    velocity(X)
        Calculates the 4-velocity at given spacetime points.
    surface_flux(x)
        Calculates the radiation flux from the medium at given spacetime points.
    flux_density(x)
        Calculates the radiation flux density at given spacetime points.
    opacity(x)
        Calculates the opacity of the medium at given spacetime points.
    hit_condition(s0, s1)
        Determines where a batch of trajectories intersects the medium.
    adjust_hit(x0, x1, p0, p1, s0, s1)
        Determines a better hit coordinate and interpolated impulse between two points.
    signed_distance(x)
        Determines the signed coordinate distance to the surface of the object.
    """

    def __init__(self, spacetime: Spacetime, mass: float, temperature: float = 6_000.0):
        """
        Initializes the abstract base class for a radiating medium.

        Parameters
        ----------
        spacetime : bhtrace.geometry.Spacetime
            The spacetime in which the medium exists.
        mass : float
            The characteristic mass of the astrophysical object in SI units (kg).
        temperature : float, optional
            The characteristic temperature of the object in SI units (K).
            Defaults to 6,000.0 K.

        Notes
        -----
        The `_nu_scale`, `_bb_scale`, and `_bb_pow` attributes are derived
        from the `GRRTUnitSystem` and are used internally for scaling
        frequency and blackbody radiation calculations.
        """
        self.mass = mass
        self.spacetime = spacetime
        self.units = bhU.GRRTUnitSystem(mass=mass, temperature=temperature)
        self._nu_scale = self.units.conversion_factor(bhU.Hz)
        self._bb_scale = self.units.conversion_factor((bhU.h / bhU.c.pow(2)).to(self.units))
        self._bb_pow = self.units.conversion_factor((bhU.h / bhU.kB))

    @abstractmethod
    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the rest mass density of the medium at
        given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime
            (e.g., [t, r, theta, phi] in spherical coordinates).

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the rest mass density at
            each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the temperature of the medium at
        given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the temperature at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def pressure(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Abstract method: Calculates the pressure within the disk at given
        spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the pressure at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def velocity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the 4-velocity of the medium at given
        spacetime points.

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [..., 4] representing the 4-velocity
            (e.g., [u_t, u_r, u_theta, u_phi]) at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def sound_speed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method: Calculates the sound speed within the medium at given
        spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the sound speed at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the radiation flux from the medium at
        given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [..., 1] representing the radiation flux at
            each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def flux_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the radiation flux density of the medium
        at given spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the flux density at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to calculate the opacity of the medium at given
        spacetime points.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape [..., 4] representing points in spacetime.

        Returns
        -------
        torch.Tensor
            A tensor of shape [...] representing the opacity at each point.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to determine where a batch of trajectories intersects
        the medium.

        This method should return a boolean mask indicating which trajectories
        (or segments of trajectories) have hit the medium.

        Parameters
        ----------
        s0 : torch.Tensor
            A tensor of shape [num_rays, num_steps] representing the `signed_distance`
            at the start of each trajectory segment.
        s1 : torch.Tensor
            A tensor of shape [num_rays, num_steps] representing the `signed_distance`
            parameter at the end of each trajectory segment.

        Returns
        -------
        torch.Tensor (bool)
            A boolean tensor mask of shape [num_steps, num_rays], where True
            indicates that a hit occurred within that segment.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
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
        Determines a better hit coordinate and interpolated impulse between
        two points along a trajectory segment.

        This method is used to refine the intersection point with the medium's
        boundary and can be useful when working with accretion disks.
         
        By default, no adjustment is performed, and `x1`, `p1` are
        assumed to be the hit coordinate and impulse. This default behavior is
        natural for volumetric disks where the "hit" is considered to be
        any point within the volume.

        Parameters
        ----------
        x0 : torch.Tensor
            Position at the start of the segment (shape [..., 4]).
        x1 : torch.Tensor
            Position at the end of the segment (shape [..., 4]).
        p0 : torch.Tensor
            Momentum at the start of the segment (shape [..., 4]).
        p1 : torch.Tensor
            Momentum at the end of the segment (shape [..., 4]).
        s0 : torch.Tensor
            `signed_distance` at the start of the segment (shape [...]).
        s1 : torch.Tensor
            `signed_distance` at the end of the segment (shape [...]).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - **x_hit** (`torch.Tensor`): The adjusted position of the hit.
            - **p_hit** (`torch.Tensor`): The interpolated momentum at the hit position.
        """
        return x1, p1
    
    @abstractmethod
    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to determine the signed coordinate distance to the
        surface of the object from a given point.

        This is a coordinate distance, not a geodesic distance.

        - For quasi-flat objects (like thin disks), it typically returns
          positive in the upper plane, negative on the bottom, and `torch.nan`
          outside the cutoff radius.
        - For volumetric objects (like thick disks and tori), it typically
          returns negative if the point is inside the object.
        
        Parameters
        ----------
        x : torch.Tensor
            Coordinates in spacetime (shape [..., 4]).
        
        Returns
        -------
        torch.Tensor
            Signed distance of shape [...].

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def __repr__(self):
        """
        Returns a string representation of the Medium object.
        """
        return f"{self.__class__.__name__}(spacetime={self.spacetime.__class__.__name__}, mass={self.mass})"


MEDIUM_REGISTRY = Registry(Medium)


class Composite(Medium):
    """
    A composition of multiple `Medium` objects.

    This class is intended to represent a scenario where multiple distinct
    media are present in the spacetime.
    """

    def __init__(self, spacetime: Spacetime, mediums: Tuple[Medium]):
        """
        Initializes the Composite medium.

        Parameters
        ----------
        spacetime : bhtrace.geometry.Spacetime
            The spacetime in which the composite medium exists.
        mediums : tuple[bhtrace.medium.Medium, ...]
            A tuple of `Medium` objects that compose this composite medium.

        Raises
        ------
        NotImplementedError
            This class is under development
        """
        raise NotImplementedError
        # super().__init__(spacetime=spacetime)
        # self.mediums = mediums

    def hit(self, x: torch.Tensor):
        """
        Placeholder method for determining hits within a composite medium.

        This method is not yet implemented.
        """
        pass
