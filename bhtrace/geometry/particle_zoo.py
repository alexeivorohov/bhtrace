import torch
from .particle import Particle
from .spacetime import Spacetime

class Photon(Particle):

    def __init__(self, spacetime: Spacetime):
        '''
        Create a photon.
        No parameters required.
        '''
        super().__init__(spacetime=spacetime)
        self.mu = 0
        pass


    def Hmlt(self, X, P):

        ginv = self.spacetime.ginv(X)
        return 0.5*(ginv @ P) @ P


    def dHmlt(self, X, P, eps):

        dH = torch.zeros(4)
        dX = torch.eye(4)*eps

        H = self.Hmlt(X, P)

        dH[0] = (self.Hmlt(X + dX[0], P) - H)/eps
        dH[1] = (self.Hmlt(X + dX[1], P) - H)/eps
        dH[2] = (self.Hmlt(X + dX[2], P) - H)/eps
        dH[3] = (self.Hmlt(X + dX[3], P) - H)/eps

        return dH


    def GetNullMomentum(self, X, v):

        v_inv = torch.pow(v@v, -0.5)
        v = v*v_inv

        gX = self.spacetime.g(X)

        return gX @ torch.Tensor([v_inv, v[0], v[1], v[2]])


    def GetDirection(self, X, P):

        v = self.spacetime.ginv(X) @ P
        return v[1:]


    def MomentumNorm(self, X, P):

        ginvX_s = self.spacetime.ginv(X)[1:, 1:]
        p_spatial_norm = torch.pow(ginvX_s @ P[1:] @ P[1:], -0.5)
        P[1:] = P[1:]*p_spatial_norm
        return P

        

    def normp(self, X, P):

        pass