from abc import ABC, abstractmethod
from ..geometry import Spacetime
from ..functional import levi_civita_tensor

import torch



# In this and other files we tried to always name precomputed quantities as _F, _g and etc.

class Electrodynamics(ABC):


    def __init__(self):
        '''
        Serves as base interface for all ED models
        '''
        self.base =  None   
        self.lct4 = levi_civita_tensor(4) # e^{pquv}
        self.U = lambda X: torch.Tensor([1, 0, 0, 0])
        self.Fuv = self.__Fuv_s__
    

    @abstractmethod
    def __precompute__(self, X):
        pass


    @abstractmethod
    def __compute__(self, X):
        # Tuv
        pass


    def attach_st(self, spacetime: Spacetime)

        self.base = spacetime


    def compute(self, *args, **kwargs):

        self.__precompute__(*args, **kwargs)
        self.__compute__(*args, **kwargs)
        pass

    def __Fuv__(self):

        pass

    def __Fuv_s__(self):

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
        self.L = L
        self.L_F = L_F
        self.L_FF = L_FF
    
        pass


    def __precompute__(self, X):
        '''
        - X: torch.Tensor (4) - point in space-time
        '''

        self._E = self.E(X)
        self._B = self.B(X)
        self._U = self.U(X)
        self._Fuv = self.Fuv()

        self._gX = self.base.g(X)
        self._ginvX = self.base.ginv(X)

        self._F = 2*(self._gX @ self._B @ self._B - self._gX @ self._E @ self._E)
        self._L = self.L(self._F)
        self._L_F = self.L_F(self._F)
        self._L_FF = self.L_FF(self._F)
        
        # T^{uv}
        self._Tuv = self._L*self._ginvX + 

        pass



# todo: FG-ED class
class ED_FG(Electrodynamics):

    def __init__(self, L, L_F, L_FF):
        super.__init__()
        self.L = L
        self.L_F = L_F
        self.L_FF = L_FF

        pass


    def __compute__(self, X):

        self.

        pass