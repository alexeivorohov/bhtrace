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
        - X: torch.Tensor of shape [b, 4] - points for which to evaluate
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
        Compute connection symbols by 
        
        ### Inputs:
        - X: torch.Tensor [4] - evaluation point
        ### Outputs:
        - G: torch.Tensor [4, 4, 4] - connection symbols
        First index is contravariant, others are covariant.
        '''

        return None


    def conn_(self, X):
        '''
        Evaluate connection symbols by numerical differentiation. 
        Relies on method dg(X) in computing derivatives of the metric/

        ### Inputs:
        - X: torch.Tensor [4] - coordinates
        ### Outputs:
        - G: torch.Tensor [4, 4, 4] - connection symbols
        First index is contravariant, others are covariant.
        '''
        g_duv = self.dg(X)
        ginv_ = self.ginv(X)    

        dg0 = torch.einsum('bmd,bduv->bmuv', ginv_, g_duv)
        dg1 = torch.einsum('bmv,bduv->bmdv', ginv_, g_duv)
        dg2 = torch.einsum('bmu,bduv->bmud', ginv_, g_duv)


        return 0.5*( - dg0 + dg1 + dg2)

    
    @abstractmethod
    def crit(self, X):
        '''
        Function of "distance" to the metric pecularities
        F: X -> (0, inf)
        '''

        return None