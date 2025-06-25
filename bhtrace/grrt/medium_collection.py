from medium import Medium
from ..geometry import Spacetime

import torch


class ThinDisk(Medium):

    def __init__(self,
                 spacetime: Spacetime,
                 position : torch.Tensor = torch.zeros(4),
                 direction: torch.Tensor = torch.tensor([0, 0, 0, 1])
                 ):
        '''
        Thin disk accretion

        Input parameters:
        - position: torch.Tensor
        - direction: torch.Tensor
        - params: dict
        '''

        super().__init__(spacetime=Spacetime, coordinates=)
        self._flux_ = lambda r, phi: torch.pow(r, -3)*(1 - torch.pow(r/2, -0.5))
        self.pos = position
        self.dir = direction
   
        
    def Embedding(self, xi):
        '''
        Inputs:
        - xi: torch.Tensor of shape[..., 3] - disk t, r and phi

        Outputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime
        '''

        X = torch.Tensor(xi.shape)
        # Into cartesian?


        return xi

    
    def InvEmbedding(self, X: torch.Tensor):
        '''
        Inputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime   

        Outputs:
        - xi: torch.Tensor of shape[..., 2] - disk rho and phi
        '''

        # From cartesian?

        return None


    def Density(self, X: torch.Tensor):


        rho = 0

        return rho
    

    def U(self, xi: torch.Tensor):

        # Should be a method of ??

        U_disk = torch.zeros_like(X_sph)

        # Keplerian disk: v_phi = sqrt(GM/R)
        u_ph = lambda r: torch.pow(r, -0.5)


        U_disk[..., 3] = u_ph(X_sph[..., 1])

        return 


    def Hit(self, X):

        
        return X


    def Flux(self, xi): 

        f = 0
    
        return f
    

    def __call__(self, X):

        # xi = hit(X)
        # U = U(xi)
        # F = flux(xi)
        # flux = ...

        pass
    

class Spherical(Medium):

    def __init__(self, spacetime: Spacetime):

        super().__init__(spacetime=Spacetime)