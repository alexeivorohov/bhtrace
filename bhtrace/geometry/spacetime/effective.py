from .base import Spacetime
from .spherical import SphericallySymmetric
from ..electrodynamics import Electrodynamics

import torch


class EffGeom(Spacetime):

    def __init__(self, ED: Electrodynamics, base: Spacetime, E: callable, B: callable):
        '''
        Spherically-symmetric effective geometry for the case of the ED Electrodynamics

        ### Inputs:
        - ED: Electrodynamics model
        - base: Base spacetime
        - E: callable(X) - electric field in spherical coordinates
        - B: callable(X) - magnetic field in spherical coordinates
        '''

        self.base = base
        self.__coords__ = base.__coords__
        self.ED = ED
        self.ED.set_regime()
        self.ED.attach_fields(E, B)
        self.U = torch.tensor([1., 0., 0., 0.])

        pass

    # description in the base class
    def g(self, X):

        ginvX = self.ginv(X)

        return torch.inverse(ginvX)
    
    # description in the base class
    def ginv(self, X):

        ginvX = self.base.ginv(X)
        gX = self.base.g(X)
        self.ED.calculate(X, gX, self.U, ginvX)

        ginv = ginvX - 4*(self.ED._L_FF/self.ED._L_F).view(*X.shape[:-1], 1, 1)*self.ED._uFFv

        return ginv

    # description in the base class
    def conn(self, X):

        return self.conn_(X)

    # description in the base class
    def crit(self, X):

        c1 = self.base.crit(X)

        return c1
    

class EffgeomSimple(Spacetime):

    __coords__ = 'Spherical'

    def __init__(self, ED: 'Electrodynamics', f=None, f_r=None, E=None, B=None):
        '''
        Spherically-symmetric effective geometry for the case of the :ED: Electrodynamics

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
        self.ED.set_regime()
        self.ED.attach_fields(E, B)

        A = lambda r: self.w(r)*f(r)
        A_r = lambda r: self.w(r)

        B = lambda r: self.w(r)/f(r)
        B_r = lambda r: self.dw(r)/f(r)


        self.eff = SphericallySymmetric(A, A_r, B, B_r)

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
