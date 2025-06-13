from ..tracing import PTracer
from ..geometry import Particle
from ..functional import sph2cart, cart2sph

import torch


class lens2d:
    '''
    Example scenario.

    Draws flat p
    '''
    def __init__(self, path=None):
        '''
        Create empty 2d drawer or load from file, if path specified.
        '''
        if path != None:
            self.load(path)
    
        pass


    def calc(self,\
        particle: Particle | list,
        X0: torch.Tensor | list,
        V0: torch.Tensor | list,
        nsteps = 128,
        T=10.0,
        save_as = None
        ):
        '''
        Calculate worldlines over a parameter net

        Return:
        - particle: Particle() - particle to be traced
        - X0: 
        - V0:

        '''
        param_dict = {'particle': particle,\
                  'X0': X0, 
                  'V0': V0,
                  'T': T,
                  'nsteps': nsteps}
        
        

        self.calc()
        pass


    def calc(self, param_dict, save_as=None):

        self.param_net = self.net_generate()

        pass


    def _calc_(self, params, save_as=None):

        pass


    def traj_plot(self, cases=None):

        pass


    def _traj_plot_(self):

        pass


    def lens_plot(self, cases=None):

        pass


    def net_generate(self, param_dict):

        pass


    def save(self, path):

        pass


    def load(self, path):
        
        pass


if __name__ == '__main__':

    
    pass