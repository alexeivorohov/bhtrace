'''
This file contains description of hamiltonian ray-tracer, which uses general covariant hamiltonian equations:

    dx^{i}/dt = g^{ij}p_{j}

    dp^{i}/dt = - dH/dx^{i}


'''

import torch
import os
import pickle

from ..geometry import Spacetime, Particle
from ..functional import ODE
from .tracer import Tracer


class PTracer(Tracer):
    '''
    Hamiltonian ray-tracer. Solves general covariant hamiltonian equations:

        dx^{i}/dt = g^{ij}p_{j}

        dp^{i}/dt = - dH/dx^{i}

    Hamiltonian derivative is estimated numerically.
    '''

    def __init__(self, ode_method='Euler', eps=1e-3):

        super().__init__(ode_method=ode_method)

        self.m_param = None
        self.eps = eps
        self.name = 'PTracer'
        pass


    def evnt(self,
             t,
             X: torch.Tensor,
             P: torch.Tensor
             ):

        # TODO:
        # [ ] refactor this method
        # cr1 = self.particle.crit(XP[..., :4], XP[..., 4:])
        cr1 = torch.less(self.max_proper_t, X[..., 0])
        cr2 = torch.less(self.r_max, torch.abs(X[..., 1]))
        # integration continues while function returns false
        return cr1 + cr2
    

    def reg(self,
            t,
            X: torch.Tensor,
            P: torch.Tensor
            ):

        # TODO:
        # [ ] refactor this method
        # X = XP[..., :4]
        # P = self.particle.MomentumNorm(XP[..., :4], XP[..., 4:])
        # return torch.cat([X, P])

        return X, P


    def __term__(self,
                t,
                X: torch.Tensor,
                P: torch.Tensor
                ):

        ginvX = self.spc.ginv(XP[..., :4])
        # dX^mu = g^{mu nu} P_nu 
        # dP_nu = - partial_nu H
        dX = torch.einsum('...uv, ...u -> ...v', ginvX, XP[..., 4: ])
        dP = - self.particle.dHmlt(XP[..., :4], XP[..., 4:], self.eps)
        
        return torch.cat((dX, dP))
    

    def evaluation(self,
                    t,
                    X: torch.Tensor,
                    P: torch.Tensor
                    ):
        '''
        Evaluation method.

        
        Checks constraint violation for photons

            g^{ik} p_i p_k
        '''
        X = XP[..., :4]
        P = XP[..., 4:]
        ginvX = self.spc.ginv(X)
        
        outp = torch.einsum('...uv, ...u, ...v -> ...', ginvX, P, P)

        return outp

