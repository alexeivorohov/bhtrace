from abc import ABC, abstractmethod
from ..geometry import Spacetime
# from ..functional import levi_civita_tensor

import torch



# In this and other files precomputed quantities are always named as _F, _g and etc.

class Electrodynamics(ABC):
    '''
    Serves as base class for all ED models
    ''' 

    def __init__(self):
        
        # ED.lct4 = levi_civita_tensor(4) # e^{pquv}
        self.L = L
        self.L_F = L_F
        self.L_FF = L_FF

        self.L_G = L_G
        self.L_GG = L_G
        self.L_FG = L_FG

    def regime():

        pass

    def compute(self, *args, **kwargs):

        pass

    def attach_fields(self, E, B):
        '''
        Attach E and B fields coordinate representation to current model

        ### Inputs:
        - E: callable(X) - electric field tensor
        - B: callable(X) - magnetic field tensor

        '''

        self.E = E
        self.B = B

        pass


class ED_logic(ABC):
    '''
    Serves as base interface for computation logics of ED models
    '''
    
    def __call__(
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            U: torch.Tensor,
            ginvX: torch.Tensor = None
            ):
        '''
        Perform computations of all quantites of given ED model

        Inputs:
        - ED: Electrodynamics - the model within which to perform computations
        - X: torch.Tensor - point in spacetime
        - gX: torch.Tensor - metric tensor at this point
        - U: torch.Tensor - 4-velocity of the reference frame
        - ginvX: torch.Tensor - metrics inverse (do not needed by default)
        '''
        pass


    def Fuv(ED: Electrodynamics):
        '''
        Maxwell tensor with all upper indexes

        F^{uv}
        '''
        raise NotImplementedError


    def Fuv_E(ED: Electrodynamics):
        '''
        Maxwell tensor with all upper indexes
    
        Faster method for a case of single E field
        '''

        f1 = torch.outer(ED._E, ED._U)
        f2 = f1.T
        return f1-f1.T
    

    def Fuv_B(ED: Electrodynamics):
        '''
        Maxwell tensor with all upper indexes
    
        Faster method for a case of single B field
        '''
        # TODO: Implement this method
        f1 = torch.outer(ED._E, ED._U)
        f2 = f1.T
        return f1-f1.T
    
    
    def FumFmv(
            ED: Electrodynamics,
            gX: torch.Tensor
            ):
        '''
        Colvolution of Maxwell tensor with itself

        Inputs:
        - ED: Electrodynamics - the model within which to perform computations
        - X: torch.Tensor - point in spacetime
        '''

        return torch.einsum('...up, ...pq, ...qv->uv', ED._Fuv, gX, ED._Fuv)


class ED_F(ED_logic):

    @classmethod
    def __call__(
            cls,
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            ginvX=None
            ):

        ED._E = ED.E(X)
        ED._B = ED.B(X)

        ED._Fuv = cls.__Fuv_E__()
        
        ED._E2 = gX @ ED._E @ ED._E
        ED._B2 = gX @ ED._B @ ED._B 

        ED._F = 2*(ED._B2 - ED._E2)
        ED._L = ED.L(ED._F)
        ED._L_F = ED.L_F(ED._F)
        ED._L_FF = ED.L_FF(ED._F)
        
        # F^{ua}F^{v}_{a}
        ED._uFFv = 
        pass


# TODO: FG-ED class
class ED_FG(ED_logic):


    def compute(ED, X):

        raise NotImplementedError
        