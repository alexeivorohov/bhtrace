import torch
from .spacetime import Spacetime

class MinkowskiSph(Spacetime):

    def __init__(self):

        pass


    def g(self, X):
        
        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = torch.pow(X[:, 1], 2)
        outp[:, 3, 3] = torch.pow(X[:, 1]*torch.sin(X[:, 2]), 2)

        return outp


    def ginv(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = torch.pow(X[:, 1], -2)
        outp[:, 3, 3] = torch.pow(X[:, 1]*torch.sin(X[:, 2]), -2)

        return outp


    def conn(self, X):

        r = X[:, 1]
        th = X[:, 2]
        phi = X[:, 3]

        outp = torch.zeros([X.shape[0], 4, 4, 4])

        f_ = 1
        df_ = 1

        # t
        # outp[:, 0, 1, 0] = df_/2/f_
        outp[:, 0, 0, 1] = outp[:, 0, 1, 0]

        # r
        # outp[:, 1, 0, 0] = f_*df_/2
        outp[:, 1, 1, 1] = -outp[:, 0, 1, 0]
        outp[:, 1, 2, 2] = -r
        outp[:, 1, 3, 3] = outp[:, 1, 2, 2]*torch.sin(th)**2

        # th
        outp[:, 2, 1, 2] = 1/r
        outp[:, 2, 2, 1] = outp[:, 2, 1, 2]
        outp[:, 2, 3, 3] = -0.5*torch.sin(2*th)

        #phi
        outp[:, 3, 3, 1] = 1/r
        outp[:, 3, 1, 3] = outp[:, 3, 3, 1]
        outp[:, 3, 3, 2] = 1/torch.tan(th)
        outp[:, 3, 2, 3] = outp[:, 3, 3, 2]

        return outp


class SphericallySymmetric(Spacetime):

    def __init__(self, f, f_r):
        '''
        Class for handling spherically-symmetric spacetimes 

        ## Input:
        f - callable: metric function
        f_r - it's derivative wrt r
        '''
        self.f = f
        self.df = f_r
        
    def g(self, X):
        
        outp = torch.zeros([X.shape[0], 4, 4])

        f_ = self.f(X[:, 1])
        R2 = X[:, 1]**2

        outp[:, 0, 0] = -f_
        outp[:, 1, 1] = 1/f_
        outp[:, 2, 2] = R2
        outp[:, 3, 3] = R2 * torch.sin(X[:, 2])**2

        return outp

    def ginv(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        f_ = self.f(X[:, 1])

        outp[:, 0, 0] = -1/f_
        outp[:, 1, 1] = f_
        outp[:, 2, 2] = torch.pow(X[:, 1], -2)
        outp[:, 3, 3] = torch.pow(X[:, 1]*torch.sin(X[:, 2]), -2)

        return outp


    def conn(self, X):
        
        # X: [t, r, th, phi]
        r = X[:, 1]
        th = X[:, 2]
        phi = X[:, 3]

        outp = torch.zeros([X.shape[0], 4, 4, 4])

        f_ = self.f(r)
        df_ = self.df(r)

        # t
        outp[:, 0, 1, 0] = df_/2/f_
        outp[:, 0, 0, 1] = outp[:, 0, 1, 0]

        # r
        outp[:, 1, 0, 0] = f_*df_/2
        outp[:, 1, 1, 1] = -outp[:, 0, 1, 0]
        outp[:, 1, 2, 2] = -f_*r
        outp[:, 1, 3, 3] = outp[:, 1, 2, 2]*torch.sin(th)**2

        # th
        outp[:, 2, 1, 2] = 1/r
        outp[:, 2, 2, 1] = outp[:, 2, 1, 2]
        outp[:, 2, 3, 3] = -0.5*torch.sin(2*th)

        #phi
        outp[:, 3, 3, 1] = 1/r
        outp[:, 3, 1, 3] = outp[:, 3, 3, 1]
        outp[:, 3, 3, 2] = 1/torch.tan(th)
        outp[:, 3, 2, 3] = outp[:, 3, 3, 2]

        return outp
