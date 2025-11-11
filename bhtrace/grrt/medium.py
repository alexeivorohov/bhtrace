import torch
from abc import ABC, abstractmethod
from typing import Tuple

from bhtrace.geometry import Spacetime, Particle
from .utils import i2_r

# TODO:

# [ ] Implement factory method

# Future:
# [ ] Implement composite medium class
# [ ] How to treat intersections and opacity for composite mediums?

class Medium(ABC):

    __coords__: str = None
    '''Coordinates of the medium'''

    def __init__(self, 
                 spacetime: Spacetime,
                 anchor: torch.Tensor, 
                 direction: torch.Tensor,
                 ):

        self.spacetime = spacetime
        self.st_coords = spacetime.__coords__
        self.position = anchor
        

    def density(self, X: torch.Tensor):
        '''
        Inputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime

        Outputs:
        - rho: torch.Tensor of shape[...] - density
        '''
        
        return NotImplementedError
    

    def U(self, X_sph, g, norm_v):
        '''
        Inputs:
        - X: torch.Tensor of shape[..., 4] - position in spacetime

        Outputs:
        - U: torch.Tensor of shape[...] - local velocity (in disk coordinates)
        '''

        return NotImplementedError


    def flux(self, X):
        '''
        Inputs:
        - X: torch.Tensor of shape [..., 4] - point(s) in spacetime

        Outputs:
        - F: torch.Tensor of shape [..., 1] - radiation flux
        '''
        return NotImplementedError


    def hit(self, X, return_Xi=False):
        '''
        Inputs:
        - X: torch.Tensor of shape [..., 4] - point in spacetimes

        Outputs:
        - ?
        '''

        return NotImplementedError
    

    def __call__(self, X) -> dict:
        '''
        Performs GRRT computation along given geodesic(s) and returns full output

        Inputs:
        - X: torch.Tensor() - geodesic
        '''
        pass

#WIP
class Composite(Medium):

    def __init__(
            self,
            spacetime: Spacetime, 
            mediums: Tuple[Medium]
            ):

        raise NotImplementedError
        # super().__init__(spacetime=spacetime)
        # self.mediums = mediums

    
    def hit(self, X):
        '''
        
        '''
        
        pass

    