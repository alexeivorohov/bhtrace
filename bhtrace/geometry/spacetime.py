from abc import ABC, abstractmethod
# from typing_extensions import ParamSpecArgs

import torch

class Spacetime(ABC):
    '''
    Base class for handling different spacetimes.

    Required methods:
    - g(X): expression for metric function
    - ginv(X): expression for inverse metric function
    
    Optional methods:
    - conn(X): Connection symbols $\Gamma^p_uv$

    '''

    # Metrics
    @abstractmethod
    def g(self, X):
        '''
        Metric tensor

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric tensor at each coordinate
        '''
        pass


    @abstractmethod
    def ginv(self, X):
        '''
        Metric inverse

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric inverze at each coordinate
        '''
        pass


    # Connections:

    def dg(self, X, eps=2e-5):
        '''
        Numerical derviative of the metric

        ## Input:
        - X: torch.Tensor of shape [b, 4] - points in which to evaluate
        - eps: float (1e-5 default)

        ## Output

        - dg: torch.Tensor of shape [b, d, 4, 4]
        '''

        # gX = self.g(X)
        dgX = torch.zeros(X.shape[0], 4, 4, 4)

        dVec = torch.einsum('bi,ij->bij', torch.ones_like(X), torch.eye(4))*eps

        dgX[:, 0, :, :] = (self.g(X+dVec[:, 0, :]) - self.g(X-dVec[:, 0, :]))/eps/2
        dgX[:, 1, :, :] = (self.g(X+dVec[:, 1, :]) - self.g(X-dVec[:, 1, :]))/eps/2
        dgX[:, 2, :, :] = (self.g(X+dVec[:, 2, :]) - self.g(X-dVec[:, 2, :]))/eps/2
        dgX[:, 3, :, :] = (self.g(X+dVec[:, 3, :]) - self.g(X-dVec[:, 3, :]))/eps/2

        return dgX

    @abstractmethod
    def conn(self, X):
        '''
        Exact calculation of connection symbols, if possible.
        Output in sta
        X: torch.Tensor() - coordinates
        '''

        return None


    def conn_(self, X):
        '''
        Evaluation of connection symbols by numerical differentiation
        X: torch.Tensor() - coordinates
        '''
        g_duv = self.dg(X)
        ginv_ = self.ginv(X)    

        dg0 = torch.einsum('bmd,bduv->bmuv', ginv_, g_duv)
        dg1 = torch.einsum('bmv,bduv->bmdv', ginv_, g_duv)
        dg2 = torch.einsum('bmu,bduv->bmud', ginv_, g_duv)


        return 0.5*( - dg0 + dg1 + dg2)


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


class MinkowskiCart(Spacetime):

    def __init__(self):
        pass

    def g(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = 1
        outp[:, 3, 3] = 1

        return outp

    def ginv(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = 1
        outp[:, 3, 3] = 1

        return outp

    def conn(self, X):

        pass


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


class Kerr(Spacetime):

    def __init__(self, a: float):
        '''
        a: float - rotation parameter in units a/M
        '''
        self.a = a
        self.Dlta = lambda r: r**2 - 2*r + a**2
        self.Sgma = lambda r, th: r**2 + a**2 * torch.cos(th)**2
        self.P = lambda r, l: r**2+a**2-a*l
    
    def uR(self, r, l_s, q_s):

        outp = self.P(r, l_s) - self.Dlta(r)*((l_s-a)**2 + q_s**2)

        return outp

    def uTh(self, th, l_s, q_s):

        outp = q_s**2 - torch.cos(th)**2*(-a**2+ (l_s*torch.sin(th))**2)

        return outp
