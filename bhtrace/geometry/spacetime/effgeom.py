from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cached_property

import torch

from ._base import Spacetime, SpacetimeLocal

if TYPE_CHECKING:
    from bhtrace.physics.electrodynamics import Electrodynamics, ElectromagneticField


class EffectiveGeometry(Spacetime):
    """
    Effective geometry implementation for models, whose metrics depends only on `F` invariant.


    """

    def __init__(
        self,
        base: Spacetime,
        model: Electrodynamics,
        field: ElectromagneticField,
        scaled: bool = True,
    ):
        """
        Parameters
        ----------
        base : Base spacetime
            An instance of Spacetime class

        model : Electrodynamics
            An instance of Electrodynamics class

        field : ElectromagneticField
            An instance of ElectormagneticField class

        scaled: bool, defaults to True
            Uses scaled metric (compared to original expressions in Novello paper).
            This is more convenient for ray-tracing purposes.

        
        
        References
        ----------

        """
        self.base = base
        self._coords = base._coords
        self.model = model
        self.field = field
        if scaled:
            self._local_backend = EffectiveGeometryLocalFS
        else:
            self._local_backend = EffectiveGeometryLocalF

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x).g

    def ginv(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x).ginv

    def crit(self, x: torch.Tensor): 
        return None

    def local(self, x: torch.Tensor) -> "SpacetimeLocal":

        return self._local_backend(self, x)
    

class EffectiveGeometryLocalFS(SpacetimeLocal):

    def __init__(
        self,
        spacetime: EffectiveGeometry,
        x: torch.Tensor,
    ):
        super().__init__(spacetime, x)
        self.base_loc = spacetime.base.local(x)
        self.f_loc = spacetime.field.local(self.spacetime.model, self.base_loc, None)

    @cached_property
    def ginv(self) -> torch.Tensor:
        return (
           self.base_loc.ginv - self._xi.view(*self.batch, 1, 1) * self.f_loc.F_up
        )

    @cached_property
    def g(self) -> torch.Tensor:
        return (
            self._eta.view(*self.batch, 1, 1) * self.base_loc.g + 
            (self._xi * self._eta).view(*self.batch, 1, 1) * self.f_loc.F_dd
        )
    
    @cached_property
    def _eta(self) -> torch.Tensor:
        return (
            1 - (self.f_loc.L_FF / self.f_loc.L_F).pow(2) * (self.f_loc.F**2 + self.f_loc.G**2)
        ).pow(-1)
    
    @cached_property
    def _xi(self) -> torch.Tensor:
        return 4 * self.f_loc.L_FF / self.f_loc.L_F


class EffectiveGeometryLocalF(SpacetimeLocal):

    def __init__(
        self,
        spacetime: EffectiveGeometry,
        x: torch.Tensor,
    ):
        super().__init__(spacetime, x)
        self.base_loc = spacetime.base.local(x)
        self.f_loc = spacetime.field.local(spacetime.model, self.base_loc, None)

    @cached_property
    def g(self) -> torch.Tensor:
        return (
            self._a.view(*self.batch, 1, 1) * self.base_loc.g + 
            self._b.view(*self.batch, 1, 1) * self.f_loc.T_dd
        )
    
    @cached_property
    def ginv(self) -> torch.Tensor:
        return (
            self._c.view(*self.batch, 1, 1) * self.base_loc.ginv + 
            self._d.view(*self.batch, 1, 1) * self.f_loc.T_up
        )
    
    @cached_property
    def _a(self) -> torch.Tensor:
        return - self._b * (
            self.f_loc.L_F.pow(2) / self.f_loc.L_FF + 
            self.f_loc.L + self.f_loc.T * 0.5
        )

    @cached_property
    def _b(self) -> torch.Tensor:

        return 16 * self.f_loc.L_FF / self.f_loc.L_F * (
            (self.f_loc.F.pow(2) + self.f_loc.G.pow(2)) * self.f_loc.L_FF.pow(2) 
            + 16 * (self.f_loc.L_F + self.f_loc.F * self.f_loc.L_FF).pow(2)
        ).pow(-1)

    @cached_property
    def _c(self) -> torch.Tensor:
        return self.f_loc.L_F + self.f_loc.L * self._d
    
    @cached_property
    def _d(self) -> torch.Tensor:
        return self.f_loc.L_FF / self.f_loc.F
