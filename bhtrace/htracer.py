import torch
from .particle import Particle

class HTracer():

    def __init__(self):

        pass

    def particle_set(self, particle: Particle):
        '''
        
        '''

        self.pcle = particle
        self.spc = particle.Spacetime
    
    def __step__(self,  X: torch.Tensor, P: torch.Tensor, dt):

        dH = self.pcle.dHmlt(X, P, self.DVec, self.eps)

        dP = - dH*dt
        dX = P*dt

        P += dP
        X += dX

        return X, P

        
    def trace(self, X0, P0, eps, nsteps, dt):

        self.DVec = torch.einsum('bi,ij->bij', torch.ones_like(X0), torch.eye(4))
        self.eps = 1e-5

        X_res = torch.einsum('bi,n->nbi',X0, torch.zeros(nsteps))
        P_res = torch.einsum('bi,n->nbi',X0, torch.zeros(nsteps))

        X, P = X0, P0

        for i in range(nsteps-1):

            X, P = self.__step__(X, P, dt)

            X_res[i, :, :] = X
            P_res[i, :, :] = P


        return X_res, P_res


