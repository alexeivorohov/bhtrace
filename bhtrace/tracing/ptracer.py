'''
This file contains description of hamiltonian ray-tracer, which uses general covariant hamiltonian equations:

    dx^{i}/dt = g^{ij}p_{j}

    dp^{i}/dt = - dH/dx^{i}

'''

import torch
import os
import pickle

from bhtrace.geometry import Spacetime, Particle
from bhtrace.tracing._base import Tracer


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

    def __term__(self,
                t,
                X: torch.Tensor,
                P: torch.Tensor
                ):

        ginvX = self.spc.ginv(X)

        # dX^mu = g^{mu nu} P_nu 
        # dP_nu = - partial_nu H
        dX = torch.einsum('...uv, ...u -> ...v', ginvX, P)
        dP = - self.particle.dx_hmlt(X, P)
        
        if self.__const_dx__:
            s = torch.einsum('...u, ...u -> ...', P[..., 1:], dX[..., 1:])
            s = torch.pow(s, -0.5).unsqueeze(-1)
            dX *= s
            dP *= s

        return dX, dP
