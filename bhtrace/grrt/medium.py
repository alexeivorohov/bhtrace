import torch
from abc import ABC, abstractmethod
from typing import Tuple

from ..geometry import Spacetime, Coordinates, Particle
from .utils import i2_r


class Medium(ABC):

    # def __new__(cls, *args, **kwargs)
    

    def __init__(self, spacetime: Spacetime, coordinates: Coordinates):

        self.spacetime = spacetime
        self.coordinates = coordinates
        
        pass
    

    def density(self, X):
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

# TODO:
# [ ] Implement composite medium class
# [ ] How to treat intersections and opacity?

class Composite(Medium):

    # def __new__(cls, *args, **kwargs)

    def __init__(self,
                 spacetime: Spacetime, 
                 mediums: Tuple[Medium]
                 ):

        super().__init__(spacetime=spacetime)

        self.mediums = mediums

    
    def hit(self, X):
        '''
        
        '''
        
        pass

    