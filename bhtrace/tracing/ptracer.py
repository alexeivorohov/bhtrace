import torch
import os
import pickle

from ..geometry import Spacetime, Particle
from ..functional import ODE
from .tracer import Tracer


class PTracer(Tracer):

    def __init__(self, ode_method='Euler', eps=1e-4):

        super().__init__(ode_method=ode_method)

        self.name = 'PTracer'
        self.m_param = None
        self.eps = eps
        pass

    def evnt(self, t, XP):

        # cr1 = self.particle.crit(XP[:4], XP[4:])
        cr1 = torch.less(self.max_proper_t, XP[0])
        cr2 = torch.less(self.r_max, torch.abs(XP[1]))
        # integration continues while function returns false
        return cr1 + cr2
    

    def reg(self, t, XP):

        X = XP[:4]
        P = self.particle.MomentumNorm(X, XP[4:])
        return torch.cat([X, P])


    def __term__(self, t, XP):
    
        dX = self.spc.ginv(XP[:4]) @ XP[4: ]
        dP = - self.particle.dHmlt(XP[:4], XP[4: ], self.eps)
        
        return torch.cat((dX, dP))