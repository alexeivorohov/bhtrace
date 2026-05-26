import torch
import math
from typing import Tuple, Optional

from .spacetime._base import Spacetime
from .particle import Particle
from bhtrace.utils import net, rotate_points_cloud
from .transformation import relation_dict


class Observer:

    def __init__(
        self,
        spacetime: Spacetime,
        r: float = None,
        inclination: float = math.pi / 2,
        azimuth: float = 0.0,
        position: torch.Tensor = None,
        camera_dir: torch.Tensor = None,
        u: torch.Tensor = None,
    ):
        """
        Class which holds observer-related properties and methods.

        The observer's position and camera direction can be set up in two ways:
        1.  Using spherical coordinates (`r`, `inclination`, `azimuth`): This is the
            recommended, user-friendly method. The observer is placed at a distance `r`
            from the origin. `inclination` is the polar angle from the z-axis (e.g.,
            pi/2 for the equatorial plane), and `azimuth` is the azimuthal angle.
            The camera will, by default, point towards the origin.
        2.  Using a Cartesian `position` tensor and `camera_dir` vector: This is for
            backward compatibility or for setups requiring precise Cartesian placement.

        Assumes a Cartesian-like coordinate system for internal representation.

        Parameters
        ----------
        spacetime : Spacetime
            The spacetime object.
        r : float, optional
            The distance of the observer from the origin. If provided, spherical
            setup is used. Defaults to None.
        inclination : float, optional
            The inclination angle (polar angle theta) in radians from the z-axis.
            Defaults to pi/2 (equatorial plane). Used with `r`.
        azimuth : float, optional
            The azimuthal angle (phi) in radians. Defaults to 0. Used with `r`.
        position : torch.Tensor, optional
            Observer's 4-position (t, x, y, z). Used if `r` is not provided.
            Defaults to [0, 20, 0, 0] if `r` is also None.
        camera_dir : torch.Tensor, optional
            Spatial direction vector the camera is pointing in.
            If `r` is given, defaults to pointing at the origin.
            If `r` is not given, defaults to [-1, 0, 0].
        u : torch.Tensor, optional
            Observer's 4-velocity. Defaults to a stationary observer.
        """
        self.spacetime = spacetime

        if r is not None:
            x = r * math.sin(inclination) * math.cos(azimuth)
            y = r * math.sin(inclination) * math.sin(azimuth)
            z = r * math.cos(inclination)
            self.position = torch.tensor([0.0, x, y, z], dtype=torch.float32)
            # By default, camera points to origin if r is specified
            if camera_dir is None:
                self.camera_dir = -self.position[1:]
            else:
                self.camera_dir = camera_dir
        else:
            # Fallback to old method for backward compatibility
            if position is None:
                # Default position if neither r nor position is given
                self.position = torch.tensor([0.0, 20.0, 0.0, 0.0])
            else:
                self.position = position

            if camera_dir is None:
                # Default camera_dir if r is not specified
                self.camera_dir = torch.tensor([-1.0, 0.0, 0.0])
            else:
                self.camera_dir = camera_dir

        # Normalize camera direction
        if torch.linalg.norm(self.camera_dir) > 1e-9:
            self.camera_dir = self.camera_dir / torch.linalg.norm(self.camera_dir)

        if u is None:
            self.u = torch.tensor([1.0, 0.0, 0.0, 0.0])
        else:
            self.u = u

        self.X_net = None
        self.P_net = None

        self.__ic_method__ = "x0"

    def state(self) -> dict:
        """Returns a dictionary representing the state of the observer.

        Returns
        -------
        dict
            A dictionary containing the observer's parameters (`position`,
            `camera_dir`, `u`, and `spacetime` state).
        """
        return {
            "spacetime": self.spacetime.state(),
            "position": self.position,
            "camera_dir": self.camera_dir,
            "u": self.u,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "Observer":
        """Creates an Observer object from a state dictionary.

        Parameters
        ----------
        state : dict
            A dictionary containing the observer's state.

        Returns
        -------
        Observer
            An instance of `Observer`.
        """
        from bhtrace.geometry.spacetime._base import Spacetime

        spacetime_state = state.pop("spacetime")
        spacetime = Spacetime.from_dict(spacetime_state)

        for key in ["position", "camera_dir", "u"]:
            if key in state:
                state[key] = torch.tensor(state[key])

        return cls(spacetime=spacetime, **state)

    def generate_net(
        self,
        net_shape: str = "square",
        net_rng: Tuple[int, int] = (16, 16),
        net_size: Tuple[float, float] = (10, 10),
        net_dist: float = 0.0,
    ) -> None:
        """Generates the observer's virtual screen grid.

        This method creates a grid of points (the "screen") in 3D space, which
        serves as the starting plane for light rays.

        Parameters
        ----------
        net_shape : str, optional
            Shape of the screen grid ('square', 'circle'). Defaults to "square".
        net_rng : tuple of int, optional
            Number of points for the grid, e.g., (width, height). Defaults to (16, 16).
        net_size : tuple of float, optional
            Size of the grid in the observer's local frame. Defaults to (10, 10).
        net_dist : float, optional
            Distance of the screen from the observer's position along the camera
            direction. Defaults to 0.0.
        """
        # 1. Generate a canonical grid of points for the screen in the y-z plane.
        local_x, local_y, local_z = net(
            shape=net_shape, rng=net_rng, YZ0=[0, 0], X0=0, YZsize=net_size
        )
        local_screen_points = torch.stack([local_x, local_y, local_z], dim=-1)

        # 2. Rotate the screen so its normal aligns with the camera's direction.
        canonical_normal = torch.tensor([1.0, 0.0, 0.0])
        rotated_screen_points = rotate_points_cloud(
            local_screen_points, canonical_normal, self.camera_dir
        )

        # 3. Translate the screen to the observer's position.
        global_screen_points_spatial = (
            rotated_screen_points + self.position[1:] + self.camera_dir * net_dist
        )

        time_coord = torch.full(
            (global_screen_points_spatial.shape[0], 1), self.position[0].item()
        )
        self.X_net = torch.cat([time_coord, global_screen_points_spatial], dim=-1)

    def setup_ic(
        self, particle: Particle, vel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sets up initial conditions for tracing from the observer's screen.

        This method takes the screen grid generated by `generate_net` and calculates
        the initial 4-positions and 4-momenta for a given particle type.

        Parameters
        ----------
        particle : Particle
            The particle object (e.g., Photon) for which to set up initial conditions.
        vel : torch.Tensor, optional
            The initial 4-velocity of the particles. If None, it is assumed rays are
            parallel to the camera direction. Defaults to None.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - X_net (torch.Tensor): Initial positions.
            - P_net (torch.Tensor): Initial 4-momenta.
        """
        pos = self.X_net

        if vel is None:
            vel = torch.ones(pos.shape[0], 4)
            vel[:, 1:] = - self.camera_dir
        elif vel.shape[0] == 1 and vel.shape[1] == 4:
            vel = vel.repeat(pos.shape[0], 1)

        # Translate to particle coordinates
        if particle.__coords__ != "Cartesian":
            pos, vel = relation_dict["Cartesian"][particle.__coords__]().tensor(
                pos, vel, [True]
            )

        # Prepare null momenta and translate to given dtype
        self.X_net = pos.to(dtype=torch.float32)
        self.P_net = particle.null_momentum(pos, vel).to(dtype=torch.float32)

        return self.X_net, self.P_net
