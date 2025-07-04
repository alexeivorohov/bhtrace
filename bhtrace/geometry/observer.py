import torch

from . import Spacetime, Particle
from ..functional import net, EulerRotation

import torch


class Observer:

    def __init__(self,
                 spacetime: Spacetime,
                 position: torch.Tensor = torch.Tensor([0, 20, 0, 0, 0]),
                 camera_dir: torch.Tensor = torch.Tensor([0, -1, 0, 0]),
                 u: torch.Tensor = torch.Tensor([1, 0, 0 ,0]),
                 ):
        '''
        Class, which holds observer-related properties and methods
        '''
        
        self.spacetime = spacetime
        self.position = position
        self.dir = camera_dir
        self.u = u
        self.X_net = None
        
    
    def setup_net(self, *args):

        self.X_net = net(*args)

        pass


    def position_net(self, *args):
        '''
        Generate and position net, prepare particle impulses
        '''

        xyz = net(*args)        

        pass

    
    def setup_geod_ic(self, particle: Particle, *args):
        
        P = None

        self.P_net = particle.GetNullMomentum(self.X_net, self.P_net)
    

    


    