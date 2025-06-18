from .coordinates import Coordinates

import sympy
import torch


class Cartesian(Coordinates):

    def __init__(self, *kwargs):
        '''
        
        '''
        super().__init__(*kwargs)

    
    def x_in(self, X: torch.Tensor):

        

        return

    def x_out(self, X: torch.Tensor):
        
        return 



class Spherical(Coordinates):

    def __init__(self):
        '''
        
        '''
        super().__init__(name='spherical', dimension=4)

    



class Cylindric(Coordinates):

    def __init__(self):
        '''
        
        '''