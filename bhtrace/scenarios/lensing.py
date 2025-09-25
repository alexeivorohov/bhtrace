'''
This file contains recursive procdure for constucting high-resolution lensing curves.\
 - e.g. deflection angle vs imapct factor dependency
'''
from typing import Tuple
import torch

from bhtrace.trajectory import Trajectory
from bhtrace.tracing import Tracer
from bhtrace.geometry import Particle, Observer
from bhtrace.geometry.transformation import relation_dict
from bhtrace.utils import weightened_upsample_1d

def eval_lens(traj: Trajectory, verbose=False):

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

    _eps_ = 0.5
    '''Weightening factor for `weigthened_upsample_1d`'''
    _func_ = abs
    '''Weightening func for `weigthened_upsample_1d`'''


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

        for i in range(nsplits):

            x_new, dphi, new_mask = weightened_upsample_1d(x_new, 
                                                           dphi,
                                                           func=cls._func_,
                                                           eps=cls._eps_
                                                           )

            pos = x_new[new_mask, ...]
            vel = v0.repeat(pos.shape[0], 1)

            obs.X_net = pos
            pos, p0 = obs.setup_ic(particle=particle)

            traj_ = tracer.forward(particle=particle, X0=pos, P0=p0, T=T, nsteps=nsteps)
            
            dphi[new_mask] = eval_lens(traj_)
            trajs.append(traj_)

        traj.join(trajs)

        return x_new, dphi, traj
    
    @classmethod
    def setup(cls, config) -> None:

        if config == 'sharp':
            cls._func_ = lambda x: 5*abs(x)
            cls._eps_ = 0.1
        elif config == 'balanced':
            cls._func_ = abs
            cls._eps_ = 0.5
        elif config == 'log':
            cls._func_ = lambda x: torch.log(abs(x)+1)
            cls._eps_ = 0.5
        else:
            print(f'Unknown configuration: {config}. No changes were made.')
