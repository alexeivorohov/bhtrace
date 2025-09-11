'''
This file contains an abstract class Spacetime, which calculates the principal
quantities associated with a given spacetime: the metric and the Levi-Civita connection
symbols. The abstract class Spacetime contains the following methods:

    g(self, X): 
        Abstract method for calculating the metric at a given point X
        (or at a batch of points, feature under development).

    ginv(self, X):
        Abstract method for calculating the inverse metric at a given point X
        (with raised indices).

    dg(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses the simplest first-oder
        difference scheme.

    dg_horder(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses order 2 or order 4 difference
        scheme for calculating derivatives.

    conn(self, X):
        Abstract method. Computes the Levi-Civita connection coefficients 
        (Christoffel symbols) via a method specified down the line.

    conn_(self, X, method='standard'):
        Computes the Levi-Civita connection coefficients numerically
        using either dg ('standard') or dg_horder ('horder') method. 

    crit(self, X):
        Abstract method. Computes the "proximity" value to 
        the singularity of the metric.
'''

from abc import ABC, abstractmethod

# from typing_extensions import ParamSpecArgs

import torch

# May be derivatives and connections should be moved to another class?
# In this case, dealing with different coordinate systems may be simpler (?)

class Spacetime(ABC):
    '''
    Base class for handling different spacetimes.

    Required methods:
    - g(X): expression for metric function
    - ginv(X): expression for inverse metric function
    
    Optional methods:
    - conn(X): Connection symbols Gamma^p_uv

    '''
    __analytic_conn__ = False

    @abstractmethod
    def g(self, X):
        '''
        Metric tensor evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [..., 4] - coordinates

        ## Output
        g: torch.Tensor [..., 4, 4] - metric tensor at each coordinate
        '''
        pass

    @abstractmethod
    def ginv(self, X):
        '''
        Metric inverse evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [..., 4] - coordinates

        ## Output
        g: torch.Tensor [..., 4, 4] - metric inverse at each coordinate
        '''

        pass


    def dg(self, X, eps=2e-5) -> torch.Tensor:
        '''
        Numerical derviative of the metric

        ## Input:
        - X: torch.Tensor of shape [..., 4] - point for which to evaluate
        - eps: float (2e-5 default)

        ## Output

        - dg: torch.Tensor of shape [..., 4, 4, 4]
        '''

        gX = self.g(X)
        dgX = torch.zeros(*X.shape[:-1], 4, 4, 4)

        dVec = torch.eye(4).repeat(*X.shape[:-1], 1, 1) * eps


        dgX[..., 0, :, :] = (self.g(X + dVec[..., 0, :]) - gX) / eps
        dgX[..., 1, :, :] = (self.g(X + dVec[..., 1, :]) - gX) / eps
        dgX[..., 2, :, :] = (self.g(X + dVec[..., 2, :]) - gX) / eps
        dgX[..., 3, :, :] = (self.g(X + dVec[..., 3, :]) - gX) / eps

        return dgX


    def conn(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Levi-Civita connection coefficients (Christoffel symbols) 
        via a method specified down the line.
        
        ### Inputs:
        - X: torch.Tensor [..., 4] - evaluation point(s)

        ### Outputs:
        - G: torch.Tensor [..., 4, 4, 4] - connection symbols
        First index is contravariant, the other two are covariant.
        '''

        return self.conn_(X, method='standard')


    def conn_(self, X, method='standard'):
        '''
        Evaluate connection symbols by numerical differentiation. 
        Relies on method dg(X) in computing derivatives of the metric

        ### Inputs:
        - X: torch.Tensor [4] - coordinates
        ### Outputs:
        - G: torch.Tensor [4, 4, 4] - connection symbols
        First index is contravariant, others are covariant.
        '''

        if method == 'standard':
            g_duv = self.dg(X)
            ginv_ = self.ginv(X)
            
        elif method == 'horder':
            g_duv = self.gd_horder(X)
            ginv_ = self.ginv(X)

        dg0 = torch.einsum('...md, ...duv ->...muv', ginv_, g_duv)
        dg1 = torch.einsum('...mv,...duv -> ...mdv', ginv_, g_duv)
        dg2 = torch.einsum('...mu, ...duv -> ...mud', ginv_, g_duv)
        
        return 0.5*( - dg0 + dg1 + dg2)
    

    def tetrad(self, X: torch.Tensor, method='gd'):

        if method == 'gd':
            return self.tetrad_gd(X)
        else:
            return self.tetrad_linalg(X)


    @abstractmethod
    def crit(self, X: torch.Tensor):
        '''
        Function of "distance" to the metric singularities
        F: X -> (0, inf)

        Used to control step size or stop integration.
        '''

        return None
    

    def compile(self):
        '''
        Compile the class with torch.jit.script
        '''

        return torch.jit.script(self)


    def __str__(self):
        return self.__class__.__name__
    

class MockSpacetime(Spacetime):
    
    def __init__(self, coefs=[1.0, 2.0, 3.0, 5.0]):
        '''
        :class:`Spacetime()` implementation, used for test purposes.
        '''

        self.coefs = coefs
        pass


    def g(self, X):
        
        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - self.coefs[0]
        outp[..., 1, 1] = self.coefs[1]
        outp[..., 2, 2] = self.coefs[2]
        outp[..., 3, 3] = self.coefs[3]

        return outp


    def ginv(self, X):

        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - 1/self.coefs[0]
        outp[..., 1, 1] = 1/self.coefs[1]
        outp[..., 2, 2] = 1/self.coefs[2]
        outp[..., 3, 3] = 1/self.coefs[3]

        return outp

    
    def crit(self, X):

        outp = abs(X[...,1]-6) + abs(X[...,2]-6) + abs(X[...,3]-6)

        return outp
