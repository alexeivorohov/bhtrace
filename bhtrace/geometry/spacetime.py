"""
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
        Abstract method. Computes the (??? distance) to 
        the (??? nearest) singularity of the metric.
"""


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

    # Metric tensors 
    @abstractmethod
    def g(self, X):
        '''
        Metric tensor evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric tensor at each coordinate
        '''
        pass

    # Inverse metric tensors
    @abstractmethod
    def ginv(self, X):
        '''
        Metric inverse evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric inverze at each coordinate
        '''

        pass

    # Numerical derivatives of metric tensor:
    def dg(self, X, eps=2e-5):
        '''
        Numerical derviative of the metric

        ## Input:
        - X: torch.Tensor of shape [4] - point for which to evaluate
        - eps: float (2e-5 default)

        ## Output

        - dg: torch.Tensor of shape [4, 4, 4]
        '''

        gX = self.g(X)
        dgX = torch.zeros(4, 4, 4)

        dVec = torch.eye(4) * eps


        dgX[0, :, :] = (self.g(X + dVec[0, :]) - gX) / eps
        dgX[1, :, :] = (self.g(X + dVec[1, :]) - gX) / eps
        dgX[2, :, :] = (self.g(X + dVec[2, :]) - gX) / eps
        dgX[3, :, :] = (self.g(X + dVec[3, :]) - gX) / eps

        return dgX


    # Higher-order numerical derivative of metric
    def dg_horder(self, X: torch.Tensor, eps=2e-5, order=2) -> torch.Tensor:
        """
        Numerical derivative of the metric. Uses higher order difference schemes
        for better approximation of partial derivatives of the metric g(X). A difference 
        scheme is defined by coefficients before g(X + eps * k * dX_i) where eps is the
        approximation step, k is the number of steps, and dX_i is the i'th basis vector.
        
            dg(X)_i = [g(X + eps * k * dX_i) for k in shifts].T @ coefficients

        degree = 2:
            dg(X, i) = (g(X + eps * dX_i) - g(X - eps * dX_i)) / (2 * eps)
            shifts = [1, -1]
            coefficients = [0.5, -0.5]

        degree = 4:
            shifts = [2, 1, -1, 2]
            coefficients = [-1/12, 2/3, -2/3, 1/12]    
        
        ---
        INPUT:

            X:      4-vector, torch.Tensor of shape [4], the point
                    at which the derivative is evaluated

            eps:    approximation step, torch.float32

            order:  order of the difference scheme.
                    Two options available, 2 and 4

        ---
        RETURNS:

            dgX:    torch.Tensor of shape [4, 4, 4],
                    where index 0 is the direction in which
                    the partial derivative is taken


        ---
        RAISES:

            ValueError:     In case the order parameter
                            does not equal 2 or 4
        """

        gX = self.g(X)
        dgX = torch.zeros(4, 4, 4)
        dVec = torch.eye(4) * eps

        if order == 2:
            for i in range(4):
                dgX[i, :, :] = self.g(X + dVec[i, :]) - self.g(X - dVec[i, :]) / (2 * eps)

        elif order == 4:
            for i in range(4):
                dgX[i, :, :] = (-1 / 12) * self.g(X + 2 * dVec[i, :]) +\
                                (2 / 3) * self.g(X + dVec[i, :]) -\
                                (2 / 3) * self.g(X - dVec[i, :]) +\
                                (1 / 12) * self.g(X - 2 * dVec[i, :])

        else:
            raise ValueError("The order value is not valid")


    @abstractmethod
    def conn(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Levi-Civita connection coefficients (Christoffel symbols) 
        via a method specified down the line.
        
        ### Inputs:
        - X: torch.Tensor [4] - evaluation point

        ### Outputs:
        - G: torch.Tensor [4, 4, 4] - connection symbols
        First index is contravariant, the other two are covariant.
        '''

        return None


    def conn_(self, X, method='standard'):
        '''
        Evaluate connection symbols by numerical differentiation. 
        Relies on method dg(X) in computing derivatives of the metric/

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

        dg0 = torch.einsum('md,duv->muv', ginv_, g_duv)
        dg1 = torch.einsum('mv,duv->mdv', ginv_, g_duv)
        dg2 = torch.einsum('mu,duv->mud', ginv_, g_duv)
        
        return 0.5*( - dg0 + dg1 + dg2)

    
    @abstractmethod
    def crit(self, X: torch.Tensor):
        '''
        Function of "distance" to the metric singularities
        F: X -> (0, inf)
        '''

        return None