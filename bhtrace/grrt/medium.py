import torch
from abc import ABC, abstractmethod

from ..geometry import Spacetime, Coordinates, Particle
from .utils import i2_r


class Medium(ABC):

    def __init__(self, spacetime: Spacetime, coordinates: Coordinates):

        self.spacetime = spacetime
        self.coordinates = coordinates
        
        pass
    

    def Density(self, X):
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
        - U: torch.Tensor of shape[...] - local velocity
        '''

        return NotImplementedError


    def Flux(self, X):
        '''
        Inputs:
        - X: torch.Tensor of shape [..., 4] - point(s) in spacetime

        Outputs:
        - F: torch.Tensor of shape [..., 1] - radiation flux
        '''
        return NotImplementedError


    def Hit(self, X, return_Xi=False):
        '''
        Inputs:
        - X: torch.Tensor of shape [..., 4] - point in spacetimes

        Outputs:
        - ?
        '''

        return NotImplementedError
    

    def __call__(self, X):

        pass

# WIP
class Composite(Medium):

    def __init__(self, spacetime: Spacetime, mediums):

        super().__init__(spacetime=spacetime)

        self.mediums = mediums

    
    def Hit(self, X):
        '''
        
        '''
        # Priority and possibility of intersections must be somehow treated
        pass

    