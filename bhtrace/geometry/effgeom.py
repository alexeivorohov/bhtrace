from .spacetime import Spacetime
from .collection_sph import SphericallySymmetric, MinkowskiSph
from ..electrodynamics import Electrodynamics

import torch


class EffGeomSPH(Spacetime):

    def __init__(self, ED: Electrodynamics, f=None, f_r=None, E=None, B=None):
        '''
        Spherically-symmetric effective geometry for the case of the ED Electrodynamics

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


        A = lambda r: self.w(r)*f(r)
        A_r = lambda r: self.w(r)

        B = lambda r: self.w(r)/f(r)
        B_r = lambda r: self.dw(r)/f(r)


        self.eff = SphericallySymmetric(A, A_r, B, B_r)

        self.ED = ED
        self.ED.attach_fields(E, B)

        pass

    # description in base class
    def g(self, X):

        ginvX = self.ginv(X)

        return torch.inverse(ginvX)
    
    # description in base class
    def ginv(self, X):

        ginvX = self.base.ginv(X)
        gX = self.base.g(X)
        self.ED.compute(X, gX, ginvX)
        ginv = ginvX - 4*self.ED._L_FF/self.ED._L_F*self.ED._uFFv

        return ginv

    # description in base class
    def conn(self, X):

        return self.conn_(X)

    # description in base class
    def crit(self, X):

        c1 = self.base.crit(X)

        return c1
    

class EffgeomSimple(Spacetime):


    def __init__(self, ED: Electrodynamics, f=None, f_r=None, E=None, B=None):
        '''
        Spherically-symmetric effective geometry for the case of the ED Electrodynamics

        ### Inputs:
        - f: callable(r) - metric function
        - f_r: callable(r) - metric function derivative
        - ED: Electrodynamics - electrodynamics model
        - E: callable(X/r?) - electric field in spherical coordinates
        - B: callable(X/r?) - magnetic field in spherical coordinates
        '''
        if f == None: 
            f = lambda r: 1 - 2.0/r
            pass
        
        
        self.ED = ED
        self.ED.attach_fields(E, B)

        pass

    # complist
    def g(self, X):

        # self.compute(X, self.)
        return self.base.conn(X)

    # complist
    def w(self):
        '''
        tt-coefficent for effective metric in spherically-symmetric case

        Does not take arguments, since uses pre-computed values
        '''
        _w = 1 - 4*self.ED._L_FF/self.ED._L_F*self.ED._E2
        
        return _w
 
    def dw(self):
        '''
        Derivative of self.w w.r.t. r

        Does not take arguments, since uses pre-computed values
        '''
        _w = 1 - 4*self.ED._L_FF/self.ED._L_F*self.ED._E2
        
        dw = 0
        return dw
    

    # complist
    def u(self):
        '''
        rr-coefficent for effective metric in spherically-symmetric case

        Does not take arguments, since uses pre-computed values
        '''
        u = 0

        return u
    
    def du(self):
        '''
        Derivative of self.u w.r.t. r

        Does not take arguments, since uses pre-computed values
        '''

        du = 0

        return du


    # complist
    def crit(self, X):

        c1 = self.base.crit(X)

        return c1
 