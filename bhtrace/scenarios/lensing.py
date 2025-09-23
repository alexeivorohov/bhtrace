'''
This file contains effective procdure for constucting high-resolution lensing curves - e.g. deflection angle vs imapct factor dependency


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
                ) -> Trajectory:

        '''
        Args:
            particle:
            tracer:
            x0: torch.Tensor - initial positions (at least 2 points). Must be in cartesian coordinates.\
                New points will be added between them
            v0: torch.Tensor - particle initial 4-velocity. Must be in cartesian coordinates.\
                Currently, only same velocity for all particles is supported, so it must be of shape [4]

        Returns:
            traj: Trajectory
        '''
        # Support for velocity upsampling

        pos = x0.copy
        vel = v0.repeat(x0.shape[0], 1)

        if particle.__coords__ != 'Cartesian':
            pos, vel = relation_dict['Cartesian'][particle.__coords__]().tensor(pos, vel, [True])
        p0 = cls.particle.GetNullMomentum(pos, vel)

        traj_ = cls.tracer.forward(pos, p0, nsteps)
        dphi_ = eval_lens(traj_)
        traj = [traj_]
        dphi = [dphi_]

        for i in range(nsplits):

            x_new, _, new_mask = weightened_upsample_1d(x_new, dphi_, eps=0.5)

            pos = x_new[new_mask, ...]
            vel = v0.repeat(pos.shape[0], 1)

            if particle.__coords__ != 'Cartesian':
                pos, vel = relation_dict['Cartesian'][particle.__coords__]().tensor(pos, vel, [True])
            
            p0 = cls.particle.GetNullMomentum(pos, vel)

            traj_ = cls.tracer.forward(pos, p0)
            dphi_ = eval_lens(traj_)

            traj.append[traj_]
            dphi.append[dphi_]
        
        
        return traj, dphi
        
if __name__=='__main__':
    import uniplot as uplot
    x0 = torch.linspace(0, 2*torch.pi, 11)
    tgt = torch.sin(x0)

    for i in range(3):
        fill = lambda x, tgt: (tgt[:-1, ...] + tgt[1:,...])/2
        x0, tgt, _ = weightened_upsample_1d(x0, tgt, eps=0.2, fill=fill)
        print(x0)
        print(tgt)
        uplot.plot(tgt, x0)

    # traj = Trajectory.load('examples/data/mwe_sph_2d.traj')
    # dphi = eval_lens(traj)
    # print(dphi.shape)
    # print(dphi)s
    
    # uplot.plot(dphi, )



