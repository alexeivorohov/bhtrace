from .medium import Medium
from bhtrace.geometry import Spacetime


from typing import Type

import torch


class ThinKeplerianDisk(Medium):

    # TODO:
    # [ ] Disk orientation 
    # [ ] Disk shifts
    # TODO: 

    def __init__(self,
                 spacetime: Spacetime,
                 position: torch.Tensor = torch.zeros(4),
                 direction: torch.Tensor = torch.tensor([0, 0, 0, 1])
                 ):
        '''
        Thin keplerian disk model

        Input parameters:
        - position: torch.Tensor
        - direction: torch.Tensor
        - params: dict
        '''

        super().__init__(
            spacetime=spacetime,
            anchor=position,
            direction=direction,
            # coordinates=Axial()
            )
        

        # Keplerian disk: v_phi = sqrt(GM/R)
        # LINK
        self.u_ph = lambda r: torch.pow(r, -0.5)
        self.u_r = lambda r: torch.zeros_like(r)
        self._flux_ = lambda r, phi: torch.pow(r, -3)*(1 - torch.pow(r/2, -0.5))
   
        
    def embedding(self, xi: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
        - xi: torch.Tensor of shape[..., 4] - disk t, r, phi and z

        Outputs:
        - X: torch.Tensor of shape[..., 4] - position in given spacetime
        '''


        return self.TF(xi)

    
    def embedding(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime   

        Outputs:
        - xi: torch.Tensor of shape[..., 4] - disk t, rho, phi and z
        '''

        return self.TF.inverse(X)


    def density(self, xi: torch.Tensor):
        '''
        Inputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime   

        Outputs:
        - xi: torch.Tensor of shape[..., 4] - disk t, rho, phi and z
        '''

        rho = 0

        return rho
    

    def U(self, xi: torch.Tensor):
        '''
        Inputs:
        - xi: torch.Tensor of shape[..., 4] - position in spacetime   

        Outputs:
        - U_xi: torch.Tensor of shape[..., 4] - disk t, rho, phi and z
        '''
        
        # Should be a method of ??
        U_disk = torch.zeros(*xi.shape[:-1], 4)
        
        r = xi[..., 1]
        
        U_disk[..., 3] = self.u_ph(r)

        return X


    def hit(self, xi: torch.Tensor):
        '''
        Inputs:
        - xi: torch.Tensor of shape[..., 4] - position in spacetime   

        Outputs:
        - U_xi: torch.Tensor of shape[..., 4] - disk t, rho, phi and z
        '''
        
        return X


    def flux(self, xi: torch.Tensor) -> torch.Tensor: 

        f = 0
    
        return f
    

    def __call__(self, X: torch.Tensor):

        
        # xi = hit(X)
        # U = U(xi)
        # F = flux(xi)
        # flux = ...

        pass
    

class SphericalAccretion(Medium):

    def __init__(self, spacetime: Spacetime):

        super().__init__(spacetime=spacetime)
