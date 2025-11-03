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
from bhtrace.geometry.transformation import relation_dict
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

    _eps_: float = 0.5
    '''Weightening factor for `weigthened_upsample_1d`'''

    _w_func_ = lambda x: 2.0*abs(x)
    '''Weightening func for `weigthened_upsample_1d`'''

    _v_func_ = None
    '''Sampling function for velocities'''

    diff_threshold = 0.1
    diff_threshold_func = lambda tgt: torch.greater(tgt, 0.1*torch.pi*2)
    '''Condition on target difference'''
    mean_threshold_func = lambda tgt: torch.greater(tgt, 0.75*torch.pi*2)
    '''Condition on target mean'''
    _eps_ = 0.1

    @classmethod 
    def forward(cls,
                particle: Particle,
                tracer: Tracer,
                x0: torch.Tensor, 
                v0: torch.Tensor,
                nsplits: int = 5,
                T: float = 30,
                nsteps: int = 128,
                ) -> Tuple[torch.Tensor, torch.Tensor, Trajectory]:
        '''
        Args:
            particle:
            tracer:
            x0: torch.Tensor - initial positions (at least 2 points). Must be in cartesian coordinates.\
                New points will be added between them
            v0: torch.Tensor - particle initial 4-velocity. Must be in cartesian coordinates.\
                Currently, only same velocity for all particles is supported, so it must be of shape [4]

        Returns:
            x_new: upsampled x0
            dphi: declination angles
            traj: Trajectory
        '''
        tracer.__tqdm_bar__ = False

        # Support for velocity upsampling
        x_new = x0.clone()

        obs = Observer(spacetime = particle.spacetime,
                       position = torch.Tensor([0, 20, 0, 0]),
                       camera_dir= torch.Tensor([-1, 0, 0]),
                       u = torch.Tensor([1, 0, 0, 0])
                       )
        
        obs.X_net = x0
        pos, p0 = obs.setup_ic(particle=particle)

        traj = tracer.forward(particle=particle,
                              X0=pos,
                              P0=p0,
                              T=T,
                              nsteps=nsteps)
        
        dphi = eval_lens(traj)
        trajs = []

        for i in tqdm.trange(0, nsplits):

            x_new, dphi, new_mask = weightened_upsample_1d(x_new, 
                                                           dphi,
                                                           func=cls._w_func_,
                                                           eps=cls._eps_,
                                                           diff_threshold=cls.diff_threshold,
                                                           diff_threshold_func=cls.diff_threshold_func,
                                                           mean_threshold_func=cls.mean_threshold_func,
                                                           )

            pos = x_new[new_mask, ...]
            vel = v0.repeat(pos.shape[0], 1)

            obs.X_net = pos
            pos, p0 = obs.setup_ic(particle=particle)

            traj_ = tracer.forward(particle=particle, X0=pos, P0=p0, T=T, nsteps=nsteps)
            
            dphi[new_mask] = eval_lens(traj_)
            trajs.append(traj_)

        traj.join(trajs)
        traj.lens = (dphi, x_new[..., 2])

        return x_new, dphi, traj
    
    @classmethod
    def setup(cls, config) -> None:

        if config == 'sharp':
            cls._w_func_ = lambda x: 5*abs(x)
            cls._eps_ = 0.1
        elif config == 'balanced':
            cls._w_func_ = abs
            cls._eps_ = 0.5
        elif config == 'log':
            cls._w_func_ = lambda x: torch.log(abs(x)+1)
            cls._eps_ = 0.5
        else:
            print(f'Unknown configuration: {config}. No changes were made.')

    @classmethod
    def prepare_ic(cls,
                   N_init: int = 3,
                   d0: torch.Tensor = torch.tensor([20, 0, 0]),
                   b0: Tuple[float] = (-16.0, 16.0),
                   v0: torch.Tensor = torch.Tensor([0., -1.0, 0., 0.]),
                   X_ab: Tuple[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor]:
        '''Prepare initial conditions for lensing simulation
        
        Args:
            N_init: number of seed trajectories
            d0: distance(shift) of initial positons from center
            b0: min an max impact factor
            v0: intital velocity. If callable, v0(X) should provide initial velocity for given X
            X_ab: points, defining impact factor line. If provided, b0 will be ignored, but d0 - not.
        
        Returns:
            Tuple[torch.Tensor], which consist of 3 elements:
            x0: initial posions
            v0: initial velocities
            e_b: unit vector in the direction of impact factor increase (required for plotting)
        '''
        
        if X_ab is None:
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

        # if isinstance(v0, callable):
        #     cls._v_func_ = v0

        return x0, v0, e_b
