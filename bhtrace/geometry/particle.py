# particle class description here
from abc import ABC, abstractmethod
from .spacetime import Spacetime

import torch

class Particle(ABC):
 

    def __init__(self, Spacetime: Spacetime):
        '''
        Base class for handling different particles
        '''

        self.Spacetime = Spacetime
        self.mu = None #dedicated p^mu p_mu

        self.g_ = None
        self.ginv_ = None
        self.dgX_ = None
        self.r_max = torch.Tensor([30.0])
        pass


    @abstractmethod
    def Hmlt(self, X, P):
        '''
        Returns $H(x^\mu, p^\mu)$

        Requires contravariant X and P as inputs!
        '''
        return None


    @abstractmethod
    def dHmlt(self, X, P):
        '''
        Hamiltonian gradient

        Returns $\partial^\mu H(x^u, p^u)$

        Requires contravariant X and P as inputs!
        '''
        return None


    def dHmlt_(self, X, P, eps):
        '''
        Less effective, but type-independent method of differentiating particle hamiltonian

        Input:
        - X: contravariant coordinate
        - P: impulse (same as for hamiltonian)
        '''

        dVec = torch.einsum('bi,ij->bij', torch.ones_like(X), torch.eye(4))*eps

        # H = self.Hmlt(X, P)
        dH = torch.zeros_like(X)

        dH[:, 0] = (self.Hmlt(X+dVec[:, 0, :], P) - self.Hmlt(X-dVec[:, 0, :], P))/eps/2
        dH[:, 1] = (self.Hmlt(X+dVec[:, 1, :], P) - self.Hmlt(X-dVec[:, 1, :], P))/eps/2
        dH[:, 2] = (self.Hmlt(X+dVec[:, 2, :], P) - self.Hmlt(X-dVec[:, 2, :], P))/eps/2
        dH[:, 3] = (self.Hmlt(X+dVec[:, 3, :], P) - self.Hmlt(X-dVec[:, 3, :], P))/eps/2

        return dH


    @abstractmethod
    def normp(self, X, P):
        '''
        Method of normalizing particle impulse P at coord X
        ## Input:
        - X: torch.Tensor() - coordinate
        - P: torch.Tensor() - impulse

        ## Output:
        - P: torch.Tensor() - normalized impulse
        - mu: float - norm
        - v: float - spatial velocity
        '''
        return None

    def GetNullMomentum(self, X, v):

        v_inv = torch.pow(v@v, -0.5)
        v = v*v_inv

        gX = self.spacetime.g(X)

        return gX @ torch.Tensor([v_inv, v[0], v[1], v[2]])


    def GetDirection(self, X, P):

        v = self.spacetime.ginv(X) @ P
        return v[1:]


    def MomentumNorm(self, X, P):
        
        pass



