import torch
import math

from .base import Spacetime
from bhtrace.utils import bisection

class MinkowskiSph(Spacetime):

    __analytic_conn__ = True
    __coords__ = 'Spherical'

    def __init__(self):
        '''
        Minkowski spacetime in spherical coordinates.
        '''
        super().__init__()
    
        pass

    def g(self, X):
        
        outp = torch.zeros(*X.shape, 4)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = torch.pow(X[...,1], 2)
        outp[..., 3, 3] = torch.pow(X[...,1]*torch.sin(X[...,2]), 2)

        return outp

    def ginv(self, X):

        outp = torch.zeros(*X.shape, 4)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = torch.pow(X[...,1], -2)
        outp[..., 3, 3] = torch.pow(X[...,1]*torch.sin(X[...,2]), -2)

        return outp

    def conn(self, X):

        r = X[...,1]
        th = X[...,2]

        outp = torch.zeros(*X.shape, 4, 4)

        f_ = 1
        df_ = 0

        # t
        outp[..., 0, 1, 0] = df_/2/f_
        outp[..., 0, 0, 1] = outp[...,0, 1, 0]

        # r
        # outp[:, 1, 0, 0] = f_*df_/2
        outp[..., 1, 1, 1] = -outp[..., 0, 1, 0]
        outp[..., 1, 2, 2] = -r
        outp[..., 1, 3, 3] = outp[..., 1, 2, 2]*torch.sin(th)**2

        # th
        outp[..., 2, 1, 2] = 1/r
        outp[..., 2, 2, 1] = outp[..., 2, 1, 2]
        outp[..., 2, 3, 3] = -0.5*torch.sin(2*th)

        #phi
        outp[..., 3, 3, 1] = 1/r
        outp[..., 3, 1, 3] = outp[..., 3, 3, 1]
        outp[..., 3, 3, 2] = 1/torch.tan(th)
        outp[..., 3, 2, 3] = outp[..., 3, 3, 2]

        return outp

    def crit(self, X):
        
        return abs(X[..., 1])

class SphericallySymmetric(Spacetime):

    __analytic_conn__ = True
    __coords__ = 'Spherical'

    def __init__(self, A=None, A_r=None, B=None, B_r=None):
        '''
        Class for handling spherically-symmetric spacetimes of type

        ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 dth^2 + r^2 sin^2(th) dphi^2

        If no arguments provided, Schwarzschild spacetime will be initialized.

        Parameters:
        - A: callable(r) - tt-component of the metric
        - A_r: callable(r) - it's derivative w.r.t. r
        - B: callable(r) - rr-component of the metric
        - B_r: callable(r) - it's derivative w.r.t. r
        '''

        if A == None:
            self.A = lambda r: - (1.0 - 2.0/r)
            self.A_r = lambda r: - 2.0*torch.pow(r, -2)
            self.B = lambda r: 1.0/(1.0 - 2.0/r)
            self.B_r = lambda r: 2.0*torch.pow(r, -2)*torch.pow(1.0 - 2.0/r, -2)
            self.r_h = 2.0
        elif B == None:
            self.A = lambda r: - A(r)
            self.A_r = lambda r: -A_r(r)
            self.B = lambda r: 1/A(r)
            self.B_r = lambda r: -A_r(r)/(A(r))**2    
        else:
            self.A = A
            self.A_r = A_r
            self.B = B
            self.B_r = B_r

        if A is not None:
            self.r_h = float(bisection(A, 0.0, 4.0))

        super().__init__()

        pass
    
    def g(self, X):
        
        outp = torch.zeros(*X.shape, 4)

        A_ = self.A(X[..., 1])
        B_ = self.B(X[..., 1])

        R2 = torch.pow(X[..., 1], 2)

        outp[..., 0, 0] = A_
        outp[..., 1, 1] = B_
        outp[..., 2, 2] = R2
        outp[..., 3, 3] = R2 * torch.sin(X[..., 2])**2

        return outp

    def ginv(self, X):

        outp = torch.zeros(*X.shape, 4)

        A_ = self.A(X[..., 1])
        B_ = self.B(X[..., 1])

        outp[..., 0, 0] = 1/A_
        outp[..., 1, 1] = 1/B_
        outp[..., 2, 2] = torch.pow(X[..., 1], -2)
        outp[..., 3, 3] = torch.pow(X[..., 1]*torch.sin(X[..., 2]), -2)

        return outp

    def conn(self, X):
        
        # X: [t, r, th, phi]
        r = X[..., 1]
        th = X[..., 2]

        outp = torch.zeros(*X.shape, 4, 4)

        A_ = self.A(r)
        dA_ = self.A_r(r)

        B_ = self.B(r)
        dB_ = self.B_r(r)

        # t
        outp[..., 0, 1, 0] = dA_/2/A_
        outp[..., 0, 0, 1] = outp[..., 0, 1, 0]

        # r
        outp[..., 1, 0, 0] = dA_/2/B_
        outp[..., 1, 1, 1] = dB_/2/B_
        outp[..., 1, 2, 2] = -r/B_
        outp[..., 1, 3, 3] = outp[..., 1, 2, 2]*torch.sin(th)**2

        # th
        outp[..., 2, 1, 2] = 1/r
        outp[..., 2, 2, 1] = outp[..., 2, 1, 2]
        outp[..., 2, 3, 3] = -0.5*torch.sin(2*th)

        # phi
        outp[..., 3, 3, 1] = 1/r
        outp[..., 3, 1, 3] = outp[..., 3, 3, 1]
        outp[..., 3, 3, 2] = 1/torch.tan(th)
        outp[..., 3, 2, 3] = outp[..., 3, 3, 2]

        return outp

    def crit(self, X):

        return abs(self.A(X[..., 1]))

class KerrBL(Spacetime):

    __coords__ = 'Spherical'
    # Coords = "BoyerLindquist"

    def __init__(self, a=0.6):

        self.a = a
        '''Dimensionless spin parameter'''

        self.a2 = a**2
        self.r_h = 1 + math.sqrt(1 - self.a2)
        self.__labels__ = ['t', 'r', '\\theta', '\\phi']
        super().__init__()

    def g(self, X):

        g = torch.zeros(*X.shape, 4)

        r = X[..., 1]
        r2 = torch.pow(r, 2)
        
        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        rho2 = r2 + self.a2*costh2
        z = 2*r/rho2
        dlta = r2 + self.a2 - 2*r
        sgma = (r2 + self.a2)**2 - self.a2*dlta*sinth2

        g[..., 0, 0] = z - 1
        g[..., 0, 3] = -z*self.a*sinth2
        g[..., 3, 0] = g[..., 0, 3]

        g[..., 1, 1] = rho2/dlta
        g[..., 2, 2] = rho2
        g[..., 3, 3] = sgma*sinth2/rho2

        return g
    
    def ginv(self, X):

        ginv = torch.zeros(*X.shape, 4)

        r = X[..., 1]
        r2 = torch.pow(r, 2)
        
        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        rho2 = r2 + self.a2*costh2
        z = 2*r/rho2
        dlta = r2 + self.a2 - 2*r
        sgma = (r2 + self.a2)**2 - self.a2*dlta*sinth2

        xi = sgma*(1-z)+rho2*self.a2*z**2*sinth2

        ginv[..., 0, 0] = - sgma/xi
        ginv[..., 0, 3] = - rho2*self.a*z/xi

        ginv[..., 1, 1] = dlta/rho2
        ginv[..., 2, 2] = 1/rho2
        ginv[..., 3, 3] = rho2*(1-z)/xi/sinth2

        return ginv

    def horizon(self, X):
        
        r = X[..., 1]
        r2 = torch.pow(r, 2)
        dlta = r2 + self.a2 - 2*r

        return dlta

class KerrNewmanBL(Spacetime):

    __coords__ = 'Spherical'
    # Coords = "BoyerLindquist"

    def __init__(self, a=0.6, q=0.4):

        self.a = a
        '''Dimensionless spin parameter'''
        self.q = q
        '''Dimensionless charge'''

        self.a2 = a**2
        self.q2 = q**2

        self.r_h = 1 + math.sqrt(1 - self.a2 - self.q2)
        self.__labels__ = ['t', 'r', '\\theta', '\\phi']
        super().__init__()

    def g(self, X):

        g = torch.zeros(*X.shape, 4)

        r = X[..., 1]
        r2 = torch.pow(r, 2)
    
        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        l2 = r2 + self.a2
        rho2 = r2 + self.a2*costh2
        dlta = 1 - 2*r/l2 + self.q2/r2
        f = l2*dlta/rho2
        xi2 = l2**2*sinth2/rho2
        
        g[..., 0, 0] = - f + self.a2*sinth2/rho2
        g[..., 0, 3] = self.a*sinth2*(f - l2/rho2)
        g[..., 3, 0] = g[..., 0, 3]

        g[..., 1, 1] = 1/f
        g[..., 2, 2] = rho2
        g[..., 3, 3] = xi2 - self.a2*sinth2**2

        return g
    
    def ginv(self, X):

        g = torch.zeros(*X.shape, 4)

        r = X[..., 1]
        r2 = torch.pow(r, 2)
    
        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        l2 = r2 + self.a2
        rho2 = r2 + self.a2*costh2
        dlta = 1 - 2*r/l2 + self.q2/r2
        f = l2*dlta/rho2
        xi2 = l2**2*sinth2/rho2

        g[..., 0, 0] = - f + self.a2*sinth2/rho2
        g[..., 0, 3] = self.a*sinth2*(f - l2/rho2)

        g[..., 1, 1] = f
        g[..., 2, 2] = 1/rho2
        g[..., 3, 3] = xi2 - self.a2*sinth2**2

        subdet = g[..., 0, 0]*g[..., 3, 3] - g[..., 0, 3]**2

        g[..., 0, 0] = g[..., 0, 0]/subdet
        g[..., 3, 3] = g[..., 3, 3]/subdet
        g[..., 0, 3] = g[..., 0, 3]/subdet
        g[..., 3, 0] = g[..., 0, 3]

        return g

    def horizon(self, X):
        
        r = X[..., 1]
        r2 = torch.pow(r, 2)
        dlta = r2 - 2*r + self.a2 + self.q2

        return dlta
