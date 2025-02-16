from abc import ABC, abstractmethod
from ..geometry import Spacetime
# from ..functional import levi_civita_tensor

import torch



# In this and other files we tried to always name precomputed quantities as _F, _g and etc.

class Electrodynamics(ABC):


    def __init__(self):
        '''
        Serves as base interface for all ED models
        ''' 
        # self.lct4 = levi_civita_tensor(4) # e^{pquv}
        self.U = lambda X: torch.Tensor([1, 0, 0, 0])
        self.Fuv = self.__Fuv_s__


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


    # F^{uv}
    def __Fuv__(self):
        '''
        Maxwell tensor with all upper indexes
        '''

        pass

    # faster method for a case B=0
    def __Fuv_s__(self):
        '''
        Maxwell tensor with all upper indexes
    
        Faster method for a case of single E field
        '''

        f1 = torch.outer(self._E, self._U)
        f2 = f1.T
        return f1-f1.T


class ED_F(Electrodynamics):

    def __init__(self, L=None, L_F=None, L_FF=None):
        '''
        Serves as base class for L(F) ED models
        Holds all general routines.

        Implementation should provide:
        - L_F: callable(F) - derivative of Lagrangian w.r.t. invariant F
        - L_FF: callable(F) - second derivatife of L w.r.t. ivariant F
        '''
        super().__init__()
        # self.L = L
        # self.L_F = L_F
        # self.L_FF = L_FF
        
    
        pass


    def compute(self, X, gX, ginvX):
        '''
        - X: torch.Tensor (4) - point in space-time
        '''

        self._E = self.E(X)
        self._B = self.B(X)
        self._U = self.U(X)
        self._Fuv = self.Fuv()
        
        self._E2 = gX @ self._E @ self._E
        self._B2 = gX @ self._B @ self._B 


        self._F = 2*(self._B2 - self._E2)
        self._L = self.L(self._F)
        self._L_F = self.L_F(self._F)
        self._L_FF = self.L_FF(self._F)
        
        # F{ua}F{v, a}
        self._uFFv = torch.einsum('up,pq,qv->uv', self._Fuv, gX, self._Fuv)

        pass



# todo: FG-ED class
class ED_FG(Electrodynamics):

    def __init__(self, L, L_F, L_G, L_FF, L_FG, L_GG):
        '''
        Electrodynamics model for case of both non-zero invariants

        ### Constructor arguments:
        - L: Lagrangian
        - L_[xy]: Lagrangian derivative wrt invarians xy:

        '''

        super().__init__()
        self.L = L
        self.L_F = L_F
        self.L_FF = L_FF

        pass


    def compute(self, X):


        pass