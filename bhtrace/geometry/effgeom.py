from .spacetime import Spacetime
from .collection_sph import SphericallySymmetric
from ..electrodynamics import Electrodynamics


class EffGeomSPH(Spacetime):

    def __init__(self, f, f_r, ED: Electrodynamics, E=None, B=None):
        '''
        Spherically-symmetric effective geometry

        ### Inputs:
        - f: callable(r) - metric function
        - f_r: callable(r) - metric function derivative
        - ED: Electrodynamics - electrodynamics model
        - E: callable(X/r?) - electric field in spherical coordinates
        - B: callable(X/r?) - magnetic field in spherical coordinates
        '''

        self.base = SphericallySymmetric(f, f_r)
        ED.attach_st(self.base)

        self.E = E
        self.B = B

        pass


    def g(self, X):

        g = self.ginv()

        pass
    

    def ginv(self, X):

        ginv = self.base.ginv(X)

        pass


    def conn(self, X):

        return self.conn_(X)


    def crit(self, X):

        c1 = self.base.crit(X)

        return c1