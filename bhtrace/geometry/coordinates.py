'''
This class is a base class for all coordinate systems.

Each coordinate system, no matter which dimension of, is meant to be embedded in absolute 4d cartesian coordinate system.

This absolute coordinate system has no physical meaning and introduced for simplicity.

'''

import torch
from abc import ABC, abstractmethod

from ..functional import EulerRotation
from typing import Tuple
# Coords are tensors of form
# X[..., 4]




class Coordinates(ABC):


    def __init__(self, name, dim=4, labels=None, position=None, direction=None):
        '''
        Base class for coordinate systems
        '''

        self.name = name
        self.dim = dim
        self.labels = labels

        if position is None:
            self.position = torch.Tensor([0, 0, 0, 0])
        else:
            self.position = position

        if direction is None:
            self.direction = torch.Tensor([0, 0, 0, 1]) 
            # is t component really needed?
        else:
            self.direction = direction


        # choice between left and right systems?
        # self.domain?


    def set(self, position=None, direction=None, update=True):
        '''
        Changes parameters of coordinate system and recompiles it's methods

        Optional inputs:
        - position: (type) position of coordinate system within the base (cartesian) system
        - direction: (type) direction of coordinate system within the base (cartesian) system
        '''

        if position is not None: self.position = position

        if direction is not None: self.direction = direction

        if update: update()

        # Definitions

        # Compile coordinate transformations

        # Compile jacobians
        
        # Compile tensor transformations?

        pass

    

    def rotate(self, dphi: torch.Tensor, dtheta, update=True):
        '''
        
        '''

        self.direction[1:] = EulerRotation(self.direction[1:])

        if update: self.update()

        pass
    

class PatchCoordinates(Coordinates):

    def __init__(self, patches, coordinates):
        '''
        Class for holding complex, composite coordinate sheets

        WIP
        '''

        raise NotImplementedError
    

