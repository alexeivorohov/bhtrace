'''
This file contains effective procdure for constucting high-resolution lensing curves.\
 - e.g. deflection angle vs imapct factor dependency
'''
from typing import Tuple
import torch

from bhtrace.trajectory import Trajectory
from bhtrace.tracing import Tracer
from bhtrace.geometry import Particle
from bhtrace.geometry.transformation import relation_dict
from bhtrace.functional import weightened_upsample_1d

def eval_lens(traj: Trajectory):

    x, p = traj['Spherical']
    
    dphi_ = abs(x[..., 1:, 3] - x[..., :-1, 3])*torch.sign(p[..., :-1, 3])
    dphi = abs(dphi_.sum(dim=1))

    return dphi

class Lensing:

    @classmethod 
    def forward(cls,
                particle: Particle,
                tracer: Tracer,
                x0: torch.Tensor, 
                v0: torch.Tensor,
                nsplits=5,
                T=30,
                nsteps=128,
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
            x_new: 
            traj: Trajectory
        '''
        # Support for velocity upsampling

        pos = x0.clone()
        x_new = x0.clone()
        vel = v0.repeat(x0.shape[0], 1)

        if particle.__coords__ != 'Cartesian':
            pos, vel = relation_dict['Cartesian'][particle.__coords__]().tensor(pos, vel, [True])
        p0 = particle.GetNullMomentum(pos, vel)

        traj = tracer.forward(particle=particle,
                              X0=pos,
                              P0=p0,
                              T=T,
                              nsteps=nsteps)
        
        dphi = eval_lens(traj)
        trajs = []

        for i in range(nsplits):

            x_new, dphi, new_mask = weightened_upsample_1d(x_new, dphi, eps=0.5)

            pos = x_new[new_mask, ...]
            vel = v0.repeat(pos.shape[0], 1)

            if particle.__coords__ != 'Cartesian':
                pos, vel = relation_dict['Cartesian'][particle.__coords__]().tensor(pos, vel, [True])
            
            p0 = particle.GetNullMomentum(pos, vel)

            traj_ = tracer.forward(particle=particle, X0=pos, P0=p0, T=T, nsteps=nsteps)
            
            dphi[new_mask] = eval_lens(traj_)
            trajs.append(traj_)

        traj.join(trajs)

        return x_new, dphi, traj