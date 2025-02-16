import torch
from .spacetime import Spacetime

class MinkowskiSph(Spacetime):

    def __init__(self):
        '''
        Minkowski spacetime in spherical coordinates.
        '''
        pass


    def g(self, X):
        
        outp = torch.zeros([4, 4])

        outp[0, 0] = -1
        outp[1, 1] = 1
        outp[2, 2] = torch.pow(X[1], 2)
        outp[3, 3] = torch.pow(X[1]*torch.sin(X[2]), 2)

        return outp


    def ginv(self, X):

        outp = torch.zeros([4, 4])

        outp[0, 0] = -1
        outp[1, 1] = 1
        outp[2, 2] = torch.pow(X[1], -2)
        outp[3, 3] = torch.pow(X[1]*torch.sin(X[2]), -2)

        return outp


    def conn(self, X):

        r = X[1]
        th = X[2]
        phi = X[3]

        outp = torch.zeros([4, 4, 4])

        f_ = 1
        df_ = 0

        # t
        # outp[:, 0, 1, 0] = df_/2/f_
        outp[0, 0, 1] = outp[0, 1, 0]

        # r
        # outp[:, 1, 0, 0] = f_*df_/2
        outp[1, 1, 1] = -outp[0, 1, 0]
        outp[1, 2, 2] = -r
        outp[1, 3, 3] = outp[1, 2, 2]*torch.sin(th)**2

        # th
        outp[2, 1, 2] = 1/r
        outp[2, 2, 1] = outp[2, 1, 2]
        outp[2, 3, 3] = -0.5*torch.sin(2*th)

        #phi
        outp[3, 3, 1] = 1/r
        outp[3, 1, 3] = outp[3, 3, 1]
        outp[3, 3, 2] = 1/torch.tan(th)
        outp[3, 2, 3] = outp[3, 3, 2]

        return outp

    def crit(self, X):
        
        return abs(X[1])


class SphericallySymmetric(Spacetime):

    def __init__(self, A=None, A_r=None, B=None, B_r=None):
        '''
        Class for handling spherically-symmetric spacetimes of type
        $$ $$

        If no arguments provided, Schwarzschild spacetime will be initialized.

        ### Inputs:
        - A: callable(r) - metric function
        - A_r: callable(r) - it's derivative w.r.t. r
        '''
        if A == None:
            self.A = lambda r: 1.0 - 2.0/r
            self.A_r = lambda r: 2.0*torch.pow(r, -2)
            self.cr_r = 2.0
        else:
            self.A = A
            self.dA = A_r

        if B == None:
            self.B 

        pass

    # description in base class   
    def g(self, X):
        
        outp = torch.zeros([4, 4])

        A_ = self.A(X[1])
        B_ = self.B(X[1])

        R2 = X[1]**2

        outp[0, 0] = A_
        outp[1, 1] = B_
        outp[2, 2] = R2
        outp[3, 3] = R2 * torch.sin(X[2])**2

        return outp

    # description in base class
    def ginv(self, X):

        outp = torch.zeros([4, 4])

        A_ = self.A(X[1])
        B_ = self.B(X[1])

        outp[0, 0] = 1/A_
        outp[1, 1] = 1/B_
        outp[2, 2] = torch.pow(X[1], -2)
        outp[3, 3] = torch.pow(X[1]*torch.sin(X[2]), -2)

        return outp


    def conn(self, X):
        
        # X: [t, r, th, phi]
        r = X[1]
        th = X[2]
        phi = X[3]

        outp = torch.zeros([4, 4, 4])

        A_ = self.A(r)
        dA_ = self.A_r(r)

        B_ = self.B(r)
        dB_ = self.B_r(r)

        # t
        outp[0, 1, 0] = dA_/2/A_
        outp[0, 0, 1] = outp[0, 1, 0]

        # r
        outp[1, 0, 0] = dA_/2/B_
        outp[1, 1, 1] = dB_/2/B_
        outp[1, 2, 2] = -r/B_
        outp[1, 3, 3] = outp[1, 2, 2]*torch.sin(th)**2

        # th
        outp[2, 1, 2] = 1/r
        outp[2, 2, 1] = outp[2, 1, 2]
        outp[2, 3, 3] = -0.5*torch.sin(2*th)

        #phi
        outp[3, 3, 1] = 1/r
        outp[3, 1, 3] = outp[3, 3, 1]
        outp[3, 3, 2] = 1/torch.tan(th)
        outp[3, 2, 3] = outp[3, 3, 2]

        return outp


    def crit(self, X):

        return abs(self.A(X[1]))

