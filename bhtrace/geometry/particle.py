# particle class description here
from abc import ABC, abstractmethod
from .spacetime import Spacetime

import torch

class Particle(ABC):

    def __init__(self, Spacetime: Spacetime):

        self.Spacetime = Spacetime

        self.dvec = torch.zeros(1)

    @abstractmethod
    def Hmlt(self, X, P):
        pass


class Photon(Particle):

    def Hmlt(self, X, P):
        '''
        Returns $H(x^\mu, p^\mu)$

        Requires contravariant X and P as inputs!
        '''

        gX = self.Spacetime.g(X)

        H = 0.5*torch.einsum('bi,bij,bj->b', P, gX, P)

        return H

    def dHmlt(self, X, P, dVec, eps):

        gX = self.Spacetime.g(X)

       

        dg0 = (self.Spacetime.g(X+dVec[:, 0, :]*eps) - gX)/eps
        dg1 = (self.Spacetime.g(X+dVec[:, 1, :]*eps) - gX)/eps
        dg2 = (self.Spacetime.g(X+dVec[:, 2, :]*eps*10) - gX)/eps/10
        dg3 = (self.Spacetime.g(X+dVec[:, 3, :]*eps*10) - gX)/eps/10

        dgX = torch.stack([dg0, dg1, dg2, dg3])
        # print(dgX.shape)

        outp = 0.5*torch.einsum('bi,dbij,bj->bd', P, dgX, P)

        return outp
