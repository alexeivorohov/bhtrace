'''
This file contains recursive procdure for constucting high-resolution lensing curves.\
 - e.g. deflection angle vs imapct factor dependency
'''
from typing import Tuple

import torch
import tqdm

from bhtrace.trajectory import Trajectory
from bhtrace.tracing import Tracer
from bhtrace.geometry import Particle, Observer
from bhtrace.utils import weightened_upsample_1d

def eval_lens(traj: Trajectory,
              e_b: torch.Tensor = torch.Tensor([0, 1, 0]),
              verbose=False,
              ):

    x, p = traj['Spherical']
    
    dphi_ = abs(x[..., 1:, 3] - x[..., :-1, 3])
    turned = abs(dphi_) > torch.pi
    if verbose:
        print(dphi_)
        print(turned)

    dphi_[turned] = abs(dphi_[turned]-2*torch.pi)

    dphi_ *= torch.sign(p[..., :-1, 3])
    dphi = abs(dphi_.sum(dim=1))

    return dphi

class Lensing:
    '''
    This class implements effective construction of trajectories for lensing scenario
    '''

    def __init__(self, tracer: Tracer, observer: Observer, particle: Particle):
        self.tracer = tracer
        self.observer = observer
        self.particle = particle
        self.tracer.__tqdm_bar__ = False

        # Default algorithm parameters
        self._eps_: float = 0.5
        self._w_func_ = lambda x: 2.0*abs(x)
        self.diff_threshold = 0.1
        self.diff_threshold_func = lambda tgt: torch.greater(tgt, 0.1*torch.pi*2)
        self.mean_threshold_func = lambda tgt: torch.greater(tgt, 0.75*torch.pi*2)

    def forward(self,
                nsplits: int = 5,
                T: float = 30,
                nsteps: int = 128,
                ) -> Tuple[torch.Tensor, torch.Tensor, Trajectory]:
        '''
        Args:
            nsplits: int - number of recursive upsampling steps.
            T: float - integration time.
            nsteps: int - number of steps per trajectory.

        Returns:
            x_new: upsampled initial positions
            dphi: calculated declination angles
            traj: Trajectory object containing all traced trajectories
        '''

        # Use initial conditions from the observer
        x_new = self.observer.X_net.clone()
        pos, p0 = self.observer.setup_ic(particle=self.particle)

        traj = self.tracer.forward(particle=self.particle,
                              X0=pos,
                              P0=p0,
                              T=T,
                              nsteps=nsteps)
        
        dphi = eval_lens(traj)
        trajs = []

        for i in tqdm.trange(0, nsplits):

            x_new, dphi, new_mask = weightened_upsample_1d(
                x_new, 
                dphi,
                func=self._w_func_,
                eps=self._eps_,
                diff_threshold=self.diff_threshold,
                diff_threshold_func=self.diff_threshold_func,
                mean_threshold_func=self.mean_threshold_func,
            )

            # Prepare initial conditions for the new upsampled points
            upsampled_pos = x_new[new_mask, ...]
            self.observer.X_net = upsampled_pos
            pos, p0 = self.observer.setup_ic(particle=self.particle)

            traj_ = self.tracer.forward(particle=self.particle, X0=pos, P0=p0, T=T, nsteps=nsteps)
            
            dphi[new_mask] = eval_lens(traj_)
            trajs.append(traj_)

        traj.join(trajs)
        traj.lens = (dphi, x_new[..., 2])

        return x_new, dphi, traj
    
    def setup(self, config) -> None:
        if config == 'sharp':
            self._w_func_ = lambda x: 5*abs(x)
            self._eps_ = 0.1
        elif config == 'balanced':
            self._w_func_ = abs
            self._eps_ = 0.5
        elif config == 'log':
            self._w_func_ = lambda x: torch.log(abs(x)+1)
            self._eps_ = 0.5
        else:
            print(f'Unknown configuration: {config}. No changes were made.')

    @staticmethod
    def prepare_ic(
                   N_init: int = 3,
                   d0: torch.Tensor = torch.tensor([20, 0, 0]),
                   b0: Tuple[float] = (-16.0, 16.0),
                   X_ab: Tuple[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Prepare initial positions for a lensing simulation screen.
        
        Args:
            N_init: number of seed trajectories.
            d0: distance (shift) of initial positons from center.
            b0: min and max impact factor.
            X_ab: points, defining impact factor line. If provided, b0 will be ignored.
        
        Returns:
            Tuple[torch.Tensor] containing:
            x0: initial posions, shape [N_init, 4].
            e_b: unit vector in the direction of impact factor increase.
        '''
        
        if X_ab is None:
            # Create a default screen along the y-axis
            x_a = torch.tensor([0.0, b0[0], 0.0])
            x_b = torch.tensor([0.0, b0[1], 0.0])
        else:
            x_a = X_ab[0]
            x_b = X_ab[1]

        db = x_b - x_a
        span = torch.linspace(0, 1, N_init)
        e_b = db/db.norm(p=1, dim=-1) 

        x0 = torch.zeros(N_init, 4)
        x0[..., 1:] = d0
        x0[..., 1:] += x_a + torch.einsum('b, i -> bi', span, db)

        return x0, e_b
