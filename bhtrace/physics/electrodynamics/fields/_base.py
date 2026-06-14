from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import cached_property

import torch

from bhtrace.utils import levi_civita_tensor, ClassRegistry
import bhtrace.utils.units as bhU
from bhtrace.geometry.spacetime._base import Spacetime, SpacetimeLocal
from bhtrace.physics.electrodynamics.models import Electrodynamics

_lct4 = levi_civita_tensor(4)

class ElectromagneticField(ABC):
    """Represents certain configuration of the electromagnetic field"""

    def __init__(
        self, 
        # electrodynamics: Electrodynamics
    ):
        pass

    @abstractmethod
    def E(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def B(self, x: torch.Tensor) -> torch.Tensor: ...

    def local(
        self, model: Electrodynamics, st_state: SpacetimeLocal, u: torch.Tensor
    ) -> 'ElectromagneticLocal':
        return ElectromagneticLocal(model, self, st_state, u)


class ElectromagneticLocal:
    """Data transfer object for electromagnetic field properties at a point X.
    
    Binds the methods of given `ElectromagneticField` and `Electrodynamics` instances
    to a given batch of points and spacetime snapshot.

    All properties are computed once on demand and can be acessed in any time in any order.

    Notes
    -----
    This implementation assumes 
    """

    def __init__(
        self,
        model: Electrodynamics,
        field: ElectromagneticField,
        st_local: SpacetimeLocal,
        u: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------

        model : Electrodynamics
            Electrodynamics model
        field : ElectromagneticField
        
        st : SpacetimeLocal

        u : torch.Tensor

        """
        self.model = model
        self.field = field
        self.metric = st_local
        self.gX = st_local.g
        self.x = st_local.x
        self.device = self.x.device
        self.dtype = self.x.dtype
        self.batch = st_local.batch
        if u is None:
            u = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(*self.batch, 1)
        self.u = u.to(dtype=st_local.x.dtype, device=st_local.x.device)
        self.U_d = torch.einsum("...u, ...ud -> ...d", self.u, self.gX)
        self._invmroot_u2 = (
            - torch.einsum("...a, ...a -> ...", self.u, self.U_d)
        ).rsqrt().unsqueeze(-1)
        self.u = self.u * self._invmroot_u2
        self.U_d = self.U_d * self._invmroot_u2

    @cached_property
    def E(self) -> torch.Tensor:
        return self.field.E(self.x)

    @cached_property
    def B(self) -> torch.Tensor:
        return self.field.B(self.x)

    @cached_property
    def E2(self) -> torch.Tensor:
        return torch.einsum(
            "...u, ...uv, ...v -> ...",
            self.E, self.gX, self.E
        )

    @cached_property
    def B2(self) -> torch.Tensor:
        return torch.einsum(
            "...u, ...uv, ...v -> ...",
            self.B, self.gX, self.B
        )

    @cached_property
    def F(self) -> torch.Tensor:
        """Electromagnetic field invariant"""
        return 2.0 * (self.B2 - self.E2)

    @cached_property
    def G(self) -> torch.Tensor:
        """Electromagnetic field psuedo-invariant"""
        return 4 * torch.einsum(
            "...u, ...uv, ...v -> ...",
            self.E, self.gX, self.B
        )

    @cached_property
    def L(self) -> torch.Tensor:
        return self.model.L(self.F, self.G)

    @cached_property
    def L_F(self) -> torch.Tensor:
        return self.model.L_F(self.F, self.G)

    @cached_property
    def L_G(self) -> torch.Tensor:
        return self.model.L_G(self.F, self.G)

    @cached_property
    def L_FF(self) -> torch.Tensor:
        return self.model.L_GG(self.F, self.G)

    @cached_property
    def L_FG(self) -> torch.Tensor:
        return self.model.L_FG(self.F, self.G)
    
    @cached_property
    def L_GG(self) -> torch.Tensor:
        return self.model.L_GG(self.F, self.G)

    @cached_property
    def eps4_up(self) -> torch.Tensor:
        """
        Covariantly-constant levi-civita antisymmetric tensor with all upper indexes
        """
        return (
            _lct4.repeat(*self.batch, 1, 1, 1, 1).to(self.device, self.dtype) /
            self.metric.sqrtmg.view(*self.batch, 1, 1, 1, 1)
        )

    @cached_property
    def F_up(self) -> torch.Tensor:
        """
        Maxwell tensor with all upper indexes

        F^{uv}
        """
        return self.F_up_electric + self.F_up_magnetic
    
    @cached_property
    def dual_F_up(self) -> torch.Tensor:
        """
        Dual Maxwell tensor with all upper indexes
        """

        magnetic = torch.einsum('...a, ...b', self.u, self.B)
        magnetic = magnetic - magnetic.swapaxes(-1, -2)
        electric = torch.einsum('...abc, ...c', self.eta_uud, self.E)

        return magnetic + electric
    
    @cached_property
    def F_up_electric(self):
        """
        Electric part of Maxwell tensor with all upper indexes

        """
        outp = torch.einsum("...p,...q -> ...pq", self.u, self.E)
        return outp - outp.swapaxes(-1, -2)

    @cached_property
    def F_up_magnetic(self):
        """
        Magnetic part of Maxwell tensor with all upper indexes

        """
        return torch.einsum(
            "...pqu, ...u -> ...pq", 
            self.eta_uud, self.B
        )

    @cached_property
    def eta_uud(self) -> torch.Tensor:
        """ 
        Projection tensor


        """
        return torch.einsum(
            "...a, ...abcp, ...pd -> ...bcd",
            self.U_d, self.eps4_up, self.gX, 
        )
    
    @cached_property
    def F_ud(self) -> torch.Tensor:
        r"""
        Mixed-index Maxwell tensor

        :math:`F^{\mu}_{.\nu}`
        
        """
        return torch.einsum(
            "...uq, ...qd -> ...ud", 
            self.F_up, self.gX
        )


    @cached_property
    def F_dd(self) -> torch.Tensor:
        r"""
        Down-index Maxwell tensor

        :math:`F_{\mu\nu}`

        """
        return torch.einsum(
            "...qw, ...qd -> ...dw", 
            self.F_ud, self.gX,
        )


    @cached_property
    def FF_up(self) -> torch.Tensor:
        """
        Colvolution of Maxwell tensor with itself

        """
        return torch.einsum(
            "...up, ...pq ->...uq", 
            self.F_ud, self.F_up,
        )

    @cached_property
    def FF_dd(self) -> torch.Tensor:
        """
        Colvolution of Maxwell tensor with itself
        (lower indexes)

        F^{um}F^{v}_{m}
        """
        return torch.einsum(
            "...up, ...pq ->...uq",
            self.F_dd, self.F_ud
        )

    # TODO: per-model optimization
    @cached_property
    def T_up(self) -> torch.Tensor:
        """
        Energy-momentum tensor of electromagnetic field with all lower indexes
        """
        return - (
            4 * self.L_F.view(*self.batch, 1, 1) * self.FF_up +
            (self.L - self.G * self.L_G).view(*self.batch, 1, 1) * self.metric.ginv
        )

    @cached_property
    def T_ud(self) -> torch.Tensor:
        """
        Energy-momentum tensor with last lower index
        """
        return torch.einsum(
            "...ab, ...bc -> ...ac",
            self.T_up, self.metric.g
        )
    
    @cached_property
    def T_dd(self) -> torch.Tensor:
        """
        Energy-momentum tensor with last lower index
        """
        
        return - (
            4 * self.L_F.view(*self.batch, 1, 1) * self.FF_dd +
            (self.L - self.G * self.L_G).view(*self.batch, 1, 1) * self.metric.g
        )
    

    @cached_property
    def T(self) -> torch.Tensor:
        """
        Trace of energy-momentum tensor of electromagnetic field


        """
        return self.T_ud.diagonal(0, -1, -2).sum(-1)