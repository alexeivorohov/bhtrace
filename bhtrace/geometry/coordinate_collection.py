from .coordinates import Coordinates

import sympy
import torch


class Cartesian(Coordinates):
    '''
    Cartesian coordinate system
    '''
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class Spherical(Coordinates):
    '''
    Spherical coordinate system
    '''
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)



class Axial(Coordinates):
    '''
    Axial coordinate system
    '''
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
