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


class Photon(Particle):

    def __init__(self, Spacetime: Spacetime):
        '''
        Create a photon.
        No parameters required.
        '''
        self.Spacetime = Spacetime
        self.mu = 0
        pass


    def Hmlt(self, X, P):

        self.gX_ = self.Spacetime.g(X)

        H = 0.5*torch.einsum('bi,bij,bj->b', P, self.gX_, P)

        return H


    def dHmlt(self, X, P, dVec, eps):

        self.dgX_ = self.Spacetime.dg(X, eps=eps)
        self.ginv_ = self.Spacetime.ginv(X)

        outp = 0.5*torch.einsum('bmd,bi,dbij,bj->bm', self.ginv_, P, self.dgX_, P)

        return outp


    # Problems when g_0k != 0
    def normp(self, X, P):

        gX = self.Spacetime.g(X)

        V = P[:, 1:3]
        gs = gX[:, 1:3, 1:3] 
        g00 = gX[:, 0, 0]
        v2 = torch.einsum('bi,bij,bj->b', V, gs, V)
        v_inv = torch.pow(v2, -0.5)

        P_ = torch.zeros_like(P)
        P_[:, 0] = torch.pow(-g00, -0.5)
        P_[:, 1] = P[:, 1] * v_inv
        P_[:, 2] = P[:, 2] * v_inv
        P_[:, 3] = P[:, 3] * v_inv

        return P_
