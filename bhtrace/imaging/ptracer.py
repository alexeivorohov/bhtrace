import torch
import os
import pickle
import torchode as tode

from ..geometry import Spacetime, Particle

class PTracer():

    def __init__(self, r_max=40, e_tol=0.1):

        self.solv = 'PTracer'
        self.m_param = None

        self.Ni = 0
        self.Nt = 0
        self.t = 0

        self.X = None
        self.P = None
        self.X0 = None
        self.P0 = None
        self.r_max = r_max
        self.e_tol = e_tol

        pass


    def particle_set(self, particle: Particle):
        '''
        Attach class of particles to be traced

        ## Input:
        particle: Particle        
        '''

        self.particle = particle
        self.spc = particle.spacetime
        

    def step_size(self, X, P, gX, dgX):


        pass

    def evnt_check(self, X, P):

        fwd0 = torch.greater(abs(self.spc.r-self.spc.cr_r), self.e_tol)
        fwd1 = torch.less(self.spc.r, self.r_max)
        fwd2 = torch.all(torch.less(abs(P[1:]), 3))

        return fwd0*fwd1*fwd2
    


    def __term__(self, XP):

        
        X, P = []
    
        dX = 
        dP
        P += - dt * self.particle.dHmlt(X, P, eps)
        X +=  dt * self.spc.ginv(X) @ P

        return dXP


    def trace(self, X0, P0, eps=1e-3, nsteps=128, dt=0.15):
        '''
        
        '''
        self.X0 = X0
        self.P0 = P0
        self.Nt = nsteps
        self.Ni = X0.shape[0]

        self.X = torch.zeros(nsteps, self.Ni, 4)
        self.P = torch.zeros(nsteps, self.Ni, 4)

        self.X[0, :, :] = X0
        self.P[0, :, :] = P0
        

        for n in range(self.Ni):

            X, P = X0[n, :], P0[n, :]

            for i in range(nsteps-1):

                X, P = self.__step__(X, P, dt=dt, eps=eps)

                if self.evnt_check(X, P):
                    self.X[i+1, n, :] = X
                    self.P[i+1, n, :] = P
                else:
                    self.X[i+1, n:, :] = X
                    self.P[i+1, n:, :] = P
                    break


        return self.X, self.P
