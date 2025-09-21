import torch

from . import Spacetime, Particle
from ..functional import net, rotate_points_cloud
from .transformation_collection import relation_dict

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

        self.__ic_method__ = 'x0'

    def state_dict(self) -> dict:
        """Returns a dictionary representing the state of the observer.

        Returns:
            dict: A dictionary containing the observer's parameters.
        """
        return {
            'spacetime': self.spacetime.state_dict(),
            'position': self.position.tolist(),
            'camera_dir': self.camera_dir.tolist(),
            'u': self.u.tolist()
        }

    @classmethod
    def from_dict(cls, state: dict):
        """Creates an Observer object from a state dictionary.

        Args:
            state (dict): A dictionary containing the observer's state.

        Returns:
            An instance of `Observer`.
        """
        from bhtrace.geometry.spacetime import Spacetime
        spacetime_state = state.pop('spacetime')
        spacetime = Spacetime.from_dict(spacetime_state)
        
        for key in ['position', 'camera_dir', 'u']:
            if key in state:
                state[key] = torch.tensor(state[key])
                
        return cls(spacetime=spacetime, **state)

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
        pos = torch.cat([time_coord, global_screen_points_spatial], dim=-1)

        # 4. Calculate sample 4-momenta for parallel rays:
    
        vel = torch.ones(pos.shape[0], 4)
        vel[:, 1:] = self.camera_dir
        
        # 5. Translate to required CS:
        if particle.__coords__ != 'Cartesian':
            pos, vel = relation_dict['Cartesian'][particle.__coords__]().tensor(pos, vel, [True])

        # Prepare null momenta and translate to given dtype
        self.X_net = pos.to(dtype=torch.float32)
        self.P_net = particle.GetNullMomentum(pos, vel).to(dtype=torch.float32)

        return self.X_net, self.P_net