import torch

from . import Spacetime, Particle
from ..functional import net, rotate_points_cloud


class Observer:

    def __init__(self,
                 spacetime: Spacetime,
                 position: torch.Tensor = None,
                 camera_dir: torch.Tensor = None,
                 u: torch.Tensor = None):
        '''
        Class, which holds observer-related properties and methods.
        Assumes a Cartesian-like coordinate system for observer setup.

        Args:
            spacetime: The spacetime object.
            position: torch.Tensor [4] - Observer's 4-position.
                      Defaults to [0, 20, 0, 0].
            camera_dir: torch.Tensor [3] - Spatial direction vector camera is pointing in.
                        Defaults to [ -1, 0, 0].
            u: torch.Tensor [4] - Observer's 4-velocity. Defaults to stationary.
        '''
        self.spacetime = spacetime
        
        if position is None:
            self.position = torch.tensor([0., 20., 0., 0.])
        else:
            self.position = position

        if camera_dir is None:
            self.camera_dir = torch.tensor([-1., 0., 0.])
        else:
            self.camera_dir = camera_dir / torch.linalg.norm(camera_dir)

        if u is None:
            self.u = torch.tensor([1., 0., 0., 0.])
        else:
            self.u = u

        self.X_net = None
        self.P_net = None

    def setup_ic(self,
                 particle: Particle,
                 net_shape='square',
                 net_rng=(16, 16),
                 net_size=(10, 10),
                 net_dist=0.0):
        """
        Sets up the initial conditions (positions and momenta) for a grid of particles
        on the observer's virtual screen.

        This method creates a flat "screen" of points, orients it according to the
        observer's direction, places it at the observer's position, and calculates
        the initial momenta for parallel rays.

        Args:
            particle: The particle object (e.g., Photon) to trace.
            net_shape: Shape of the screen grid ('square', 'circle').
            net_rng: Number of points for the grid, e.g., (width, height).
            net_size: Size of the grid in the observer's local frame.
            net_dist: Distance of the screen from the observer's position.
        """
        # 1. Generate a canonical grid of points for the screen in the y-z plane.
        local_x, local_y, local_z = net(shape=net_shape, rng=net_rng, YZ0=[0, 0], X0=0, YZsize=net_size)
        local_screen_points = torch.stack([local_x, local_y, local_z], dim=-1)

        # 2. Rotate the screen so its normal aligns with the camera's direction.
        canonical_normal = torch.tensor([1., 0., 0.])
        rotated_screen_points = rotate_points_cloud(local_screen_points, canonical_normal, self.camera_dir)

        # 3. Translate the screen to the observer's position.
        global_screen_points_spatial = rotated_screen_points + self.position[1:] + self.camera_dir * net_dist
        
        time_coord = torch.full((global_screen_points_spatial.shape[0], 1), self.position[0].item())
        self.X_net = torch.cat([time_coord, global_screen_points_spatial], dim=-1)

        # 4. Calculate initial 4-momenta for parallel rays.
        # This simplified implementation is for Cartesian Minkowski spacetime.
        # A general implementation would need a robust GetNullMomentum method from the particle class.
        num_rays = self.X_net.shape[0]
        
        # For covariant momentum p_u in Minkowski: p_0 = -E, p_i = E*d_i
        # We set the initial energy E=1.
        p_t = -torch.ones(num_rays, 1)
        p_spatial = self.camera_dir.expand(num_rays, -1)
        
        self.P_net = torch.cat([p_t, p_spatial], dim=-1)

        # For compatibility with tracers that might expect a specific dtype
        self.X_net = self.X_net.to(dtype=torch.float32)
        self.P_net = self.P_net.to(dtype=torch.float32)