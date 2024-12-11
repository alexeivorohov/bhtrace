from .spacetime import Spacetime
from .collection_sph import SphericallySymmetric, MinkowskiSph
from ..electrodynamics import Electrodynamics

import torch


class EffGeomSPH(Spacetime):

    def __init__(self, ED: Electrodynamics, f=None, f_r=None, E=None, B=None):
        '''
        Spherically-symmetric effective geometry

        ### Inputs:
        - f: callable(r) - metric function
        - f_r: callable(r) - metric function derivative
        - ED: Electrodynamics - electrodynamics model
        - E: callable(X/r?) - electric field in spherical coordinates
        - B: callable(X/r?) - magnetic field in spherical coordinates
        '''

        if f == None:
            self.base = MinkowskiSph()
        else:
            self.base = SphericallySymmetric(f, f_r)

        self.ED = ED
        self.ED.attach_fields(E, B)

        pass


    def g(self, X):

        ginvX = self.ginv(X)

        return torch.inverse(ginvX)
    

    def ginv(self, X):

        ginvX = self.base.ginv(X)
        gX = self.base.g(X)
        self.ED.compute(X, gX, ginvX)
        ginv = ginvX - 4*self.ED._L_FF/self.ED._L_F*self.ED._uFFv

        return ginv


    def conn(self, X):

        return self.conn_(X)


    def crit(self, X):

        c1 = self.base.crit(X)

        return c1