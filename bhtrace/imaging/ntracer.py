import torch
import os
import pickle
import torchode as tode

from ..geometry import Spacetime, Particle

class NTracer():

    def __init__(self):

        self.solv = 'CTracer'
        self.m_param = None

        self.Ni = 0
        self.Nt = 0
        self.t = 0

        self.X = None
        self.P = None
        self.X0 = None
        self.P0 = None

    pass

    def particle_set(self, particle: Particle):
        '''
        Attach class of particles to be traced

        ## Input:
        particle: Particle        
        '''

        self.particle = particle
        self.spc = particle.Spacetime
        

    def step_size(self, X, P, gX, dgX):



        return dt


    def evnt_check(self, X, P):

        pass


    def __dXP__(self, X, P):
        '''
        X - contravariant
        P - covariant
        '''
        ginv_ = self.spc.ginv(X)

        dP = - self.particle.dHmlt_(X, P, eps=1e-5)
        dX = torch.einsum('buv,bu->bv', ginv_, P)

        return dX, dP


    def trace(self, X0, P0, eps, nsteps, dt):
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
        X, P = X0, P0

        for i in range(nsteps-1):

            dX, dP = self.__dXP__(X, P)

            X += dX*dt
            P += dP*dt

            self.X[i+1, :, :] = X
            self.P[i+1, :, :] = P

        return self.X, self.P
