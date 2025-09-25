from abc import ABC, abstractmethod
import inspect
from bhtrace.geometry.spacetime.base import Spacetime
from bhtrace.utils import levi_civita_tensor

import torch


class Electrodynamics(ABC):
    """Abstract base class for all nonlinear electrodynamics models.

    This class defines the interface for electrodynamics models and provides
    methods to calculate field quantities and model-specific Lagrangians.
    """
    def __new__(cls, *args, **kwargs):
        if cls is Electrodynamics:
            raise TypeError("Electrodynamics is an abstract class and cannot be instantiated directly. "
                            "Use a concrete subclass or the factory function `bhtrace.geometry.electrodynamics.create()`.")
        return super().__new__(cls)

    def __init__(self, **kwargs):
        """Initializes the Electrodynamics instance.
        """
        self.lct4 = levi_civita_tensor(4).float() 
        self.L = None
        self.L_F = None
        self.L_FF = None
        self.L_G = None
        self.L_GG = None
        self.L_FG = None

    def state(self) -> dict:
        """Returns a dictionary representing the state of the electrodynamics model.

        Returns:
            dict: A dictionary containing the model's name and parameters.
        """
        state = {'name': self.__class__.__name__}
        sig = inspect.signature(self.__class__.__init__)
        for param in sig.parameters.values():
            if param.name != 'self' and hasattr(self, param.name):
                attr = getattr(self, param.name)
                if isinstance(attr, (int, float, str, bool, torch.Tensor)):
                     state[param.name] = attr
        return state

    @classmethod
    def from_dict(cls, state: dict):
        """Creates an Electrodynamics object from a state dictionary.

        Args:
            state (dict): A dictionary containing the model's state.

        Returns:
            An instance of an `Electrodynamics` subclass.
        """
        from bhtrace.geometry.electrodynamics import create
        name = state.pop('name')
        return create(name, **state)


    def calculate(
        self,
        X: torch.Tensor,
        gX: torch.Tensor,
        U: torch.Tensor,
        ginvX: torch.Tensor = None,
    ) -> None:
        """Computes all relevant field and model quantities at a point.

        This method orchestrates the calculation of field strengths (E, B),
        the Maxwell tensor (F_uv), and the model-specific Lagrangian and its
        derivatives.

        Args:
            X (torch.Tensor): Spacetime coordinates.
            gX (torch.Tensor): Covariant metric tensor `g_uv`.
            U (torch.Tensor): Observer/frame 4-velocity `u^u`.
            ginvX (torch.Tensor, optional): Contravariant metric tensor `g^uv`.
                                             If not provided, it may be calculated
                                             if needed. Defaults to None.
        """
        self.process_fields.forward(ED=self, X=X, gX=gX, U=U, ginvX=ginvX)
        self.process_model.forward(ED=self, X=X, gX=gX, U=U, ginvX=ginvX)


    def set_regime(self, fields: str = 'B', model_type: str = 'F', verbose: bool = False):
        """Sets the computational regime based on the fields and model type.

        This method selects the appropriate low-level logic classes for
        calculating field and model quantities, optimizing for cases where
        only certain fields (E or B) or Lagrangian dependencies (F or F,G)
        are present.

        Args:
            fields (str, optional): The field configuration ('E', 'B', or 'EB').
                                    Defaults to 'B'.
            model_type (str, optional): The type of Lagrangian model ('F', 'FG', 'FJ').
                                        Defaults to 'F'.
            verbose (bool, optional): If True, prints status information.
                                      Defaults to False.
        """

        if fields == 'B':
            self.process_fields = ED_B
        elif fields == 'E':
            self.process_fields = ED_E
        elif fields == 'EB':
            self.process_fields = ED_EB
        else:
            raise ValueError
        
        if model_type == 'F':
            self.process_model = ED_F
        elif model_type == 'FG':
            self.process_model = ED_FG
        elif model_type == 'FJ':
            self.process_model = ED_FJ
        else:
            raise ValueError

        if verbose:
            print(f'nope')
        

    def process_fields(*args, **kwargs) -> None:
        '''
        Computes field stengths E and B, their norms and Maxwell tensor 
        '''

        return NotImplementedError


    def process_model(self, *args, **kwargs) -> None:
        '''
        Computes quantities, related to electrodynamics model: invariants, lagrangians, their derivatives and etc.
        '''

        return NotImplementedError


    def attach_fields(self,
                      E: callable, 
                      B: callable):
        '''
        Attach E and B fields coordinate representation to current model

        ### Inputs:
        - E: callable(X) - electric field tensor
        - B: callable(X) - magnetic field tensor

        '''
        self.E = E
        self.B = B


    def point_E(self, E):
        '''
        Electric field of point charge
        '''

        return NotImplementedError
    

    def point_B(self, B):
        '''
        Magnetic field of point charge
        '''
        return NotImplementedError


class ED_logic:
    '''
    Serves as base interface for computation logics of ED models
    '''

    def forward(
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


    def eps_mnpq(ED: Electrodynamics, gX: torch.Tensor):
        '''
        Covariantly-constant levi-civita antisymmetric tensor
        '''
        ED._sqrtmg = torch.sqrt( - torch.linalg.det(gX))
        
        ED._eps4 = ED.lct4.repeat(*gX.shape[:-2], 1, 1, 1, 1)/ED._sqrtmg.view(*gX.shape[:-2], 1, 1, 1, 1)
        

    def eta_pqu(ED: Electrodynamics, gX: torch.Tensor, U: torch.Tensor):
        '''
            eta^{pq}_u
        '''
        ED._eta4 = torch.einsum(
            '...pqwv, ...vk, ...k, ...wu -> ...pqu',
            ED._eps4, gX, U, gX)

    @classmethod
    def Fuv(cls, ED: Electrodynamics, gX, U):
        '''
        Maxwell tensor with all upper indexes

        F^{uv}
        '''
        
        return cls.Fuv_E(ED, gX, U) + cls.Fuv_B(ED)


    def Fuv_E(ED: Electrodynamics, gX: torch.Tensor,  U: torch.Tensor):
        '''
        Maxwell tensor with all upper indexes
    
        Faster method for a case of single E field
        '''
        outp = torch.einsum('...p,...q -> ...pq', ED._E, U)

        return outp - outp.mT
    

    def Fuv_B(ED: Electrodynamics):
        '''
        Maxwell tensor with all upper indexes
    
        Faster method for a case of single B field

        '''

        return torch.einsum('...pqu, ...u', ED._eta4, ED._B)
    

    def FumFmv(
            ED: Electrodynamics,
            gX: torch.Tensor
            ):
        '''
        Colvolution of Maxwell tensor with itself

        F^{um}F^{v}_{m}

        Inputs:
        - ED: Electrodynamics - the model within which to perform computations
        - gX: torch.Tensor - metrics  spacetime
        '''

        return torch.einsum(
            '...up, ...pq, ...qv->...uv', 
            ED._Fuv, gX, ED._Fuv)


    def __str__(self):
    
        pass


class ED_E(ED_logic):
    '''
    Class for performing computations in case of pure electric field
    '''

    @classmethod
    def forward(
            cls,
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            U: torch.Tensor,
            ginvX=None
            ):

        ED._E = ED.E(X)

        ED._Fuv = cls.Fuv_E(ED, gX, U)
        
        ED._E2 = gX @ ED._E @ ED._E
        ED._B2 = torch.zeros_like(ED._E2)
        ED._uFFv = cls.FumFmv(ED, gX)


class ED_B(ED_logic):
    '''
    Class for performing computations in case of pure magnetic field
    '''

    @classmethod
    def forward(
            cls,
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            U: torch.Tensor,
            ginvX=None
            ):

        ED._B = ED.B(X)

        cls.eps_mnpq(ED, gX)
        cls.eta_pqu(ED, gX, U)

        ED._Fuv = cls.Fuv_B(ED)
        
        ED._B2 = torch.einsum('...uv, ...u, ...v -> ...', gX, ED._B, ED._B )
        ED._E2 = torch.zeros_like(ED._B2)
        ED._uFFv = cls.FumFmv(ED, gX)


class ED_EB(ED_logic):
    '''
    Class for performing computations in case of both electric and magnetic field present
    '''

    @classmethod
    def forward(
        cls,
        ED: Electrodynamics,
        X: torch.Tensor,
        gX: torch.Tensor,
        U: torch.Tensor,
        ginvX = None
    ):
        
        ED._E = ED.E(X)
        ED._B = ED.B(X)

        cls.eps_mnpq(ED, gX)
        cls.eta_pqu(ED, gX, U)

        ED._Fuv = cls.Fuv(ED, gX, U)
        
        ED._E2 = gX @ ED._E @ ED._E
        ED._B2 = gX @ ED._B @ ED._B 
        ED._uFFv = cls.FumFmv(ED, gX)


class ED_F(ED_logic):
    '''
    Class for performing computations within L(F) models.
    
    '''
    @classmethod
    def forward(
            cls,
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            U: torch.Tensor,
            ginvX=None
            ):

        
        ED._F = 2*(ED._B2 - ED._E2)
        ED._L = ED.L(ED._F)
        ED._L_F = ED.L_F(ED._F)
        ED._L_FF = ED.L_FF(ED._F)
        

class ED_FG(ED_logic):
    '''
    Class for performing computations within L(F, G) models.
    
    '''

    @classmethod
    def forward(
            cls,
            ED: Electrodynamics,
            X: torch.Tensor, 
            gX: torch.Tensor, 
            U: torch.Tensor,
            ginvX=None
            ):
        
        ED._F = 2*(ED._B2 - ED._E2)
        ED._G = 4*gX @ ED._E @ ED._B

        ED._L = ED.L(ED._F, ED._G)
        ED._L_F = ED.L_F(ED._F, ED._G)
        ED._L_FF = ED.L_FF(ED._F, ED._G)

        ED._L_G = ED.L_G(ED._F, ED._G)
        ED._L_GG = ED.L_GG(ED._F, ED._G)
        ED._L_FG = ED.L_FG(ED._F, ED._G)
        

class ED_FJ(ED_logic):
    '''
    Class for performing computations within L(F, J_4) models.

    J_4 is an electrodynamic invariant F^{mn}F_{nk}F^{kl}F_{lm}, which with invariant F forms alternative basis of for invariant quantities in electrodynamics.
    
    '''

    @classmethod
    def forward(
        cls,
        ED: Electrodynamics,
        X: torch.Tensor, 
        gX: torch.Tensor,
        U: torch.Tensor,
        ginvX=None
        ):

        raise NotImplementedError
