import torch
import os
import pickle

from ..geometry import Spacetime, Particle
from ..functional import RKF23b, Euler


class PTracer():

    def __init__(self, r_max=30.0, e_tol=0.1, method='Euler'):

        self.solv = 'PTracer'
        self.m_param = None
        # self.s_param = {'':}

        self.Ni = 0
        self.Nt = 0
        self.t = 0

        if method == 'Euler':
            self.odeint = Euler()
        elif method == 'RKF23b':
            self.odeint = RKF23b()
        elif method == 'RKF23bv':
            self.odeint = RKF23b(varistep = True)
        else:
            raise NameError("ODE scheme {} is not known".format(method))

        self.X = None
        self.P = None
        self.X0 = None
        self.P0 = None
        self.r_max = r_max
        self.e_tol = e_tol


    def particle_set(self, particle: Particle):
        '''
        Attach class of particles to be traced

        ## Input:
        particle: Particle        
        '''

        self.particle = particle
        self.spc = particle.spacetime
        

    def evnt_check(self, X, P):

        fwd0 = torch.greater(abs(self.spc.r-self.spc.cr_r), self.e_tol)
        fwd1 = torch.less(self.spc.r, self.r_max)
        fwd2 = torch.all(torch.less(abs(P[1:]), 3))

        return fwd0*fwd1*fwd2


    def evnt(self, t, XP):

        cr1 = self.particle.crit(XP[:4], XP[4:])
        # cr2 = torch.greater(self.r_max, abs(XP[1]))

        return ~cr1
    

    def __term__(self, t, XP):
    
        dX = self.spc.ginv(XP[:4]) @ XP[4: ]
        dP = - self.particle.dHmlt(XP[:4], XP[4: ], self.eps)

        return torch.cat((dX, dP))


    def trace(self, X0, P0, eps=1e-3, nsteps=128, T=40):
        '''
        
        '''
        self.X0 = X0
        self.P0 = P0
        self.Nt = nsteps
        self.Ni = X0.shape[0]
        self.eps = eps

        self.X = torch.zeros(nsteps, self.Ni, 4)
        self.P = torch.zeros(nsteps, self.Ni, 4)

        self.X[0, :, :] = X0
        self.P[0, :, :] = P0

        T0 = torch.Tensor([0.0])

        for n in range(self.Ni):

            XP0 = torch.cat((X0[n], P0[n]))

            sol = self.odeint.forward(
                term=self.__term__, 
                X0=XP0, 
                T = (0.0, T),
                nsteps=nsteps,
                event_fn=self.evnt
                )

            self.X[:, n, :] = sol['X'][:, :4]
            self.P[:, n, :] = sol['X'][:, 4:]


        return self.X, self.P
