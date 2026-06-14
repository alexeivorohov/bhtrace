"""
This file describes a wrapper for medium models, which performs GRRT calculation


"""

from typing import Any, Optional, Dict, Literal, Tuple, List, Iterable, NamedTuple
from dataclasses import dataclass, field

import torch
import tqdm
import scipy.constants as constants


from bhtrace.geometry import Spacetime, Particle
from bhtrace.medium import Medium
from bhtrace.data import Trajectory, GRRTData, RunningTensor

from bhtrace.grrt.radiation import RadiativeModel


class GRRT:
    """
    Attributes
    ----------

    Methods
    -------

    compute()
        Computes GRRT model for already traced trajectories

    hook()
        A hook to pass as callback to the Tracer for computing trajectories on-the-fly

    retrieve()
        Retrieves computed grrt image.

    """

    def __init__(
        self,
        medium: Medium,
        hit_tolerance: float = 1e-3,
        linear_approx_steps: int = 1,
        skip_first: int | float = 1,
    ):
        """
        Parameters
        ----------
        medium : Medium
            The medium model to trace through.
        compute_total : bool
            Whether to compute total flux.
        frequences : Optional[Iterable[float]], optional
            Frequencies to compute spectrum for, in Hz. Defaults to None.
        lines : Optional[Iterable[float]], optional
            Line wavelengths to compute spectrum for, in meters. Defaults to None.
        hit_tolerance : float, optional
            Tolerance for medium intersection. Defaults to 1e-3.
        linear_approx_steps : int, optional
            Number of steps for linear approximation of intersection. Defaults to 1.
        skip_first : int, optional
            If passed as int - number of initial steps to skip.
            If passed as float - skip fraction of all steps at the beginning.
        """
        self.medium = medium
        self.hit_tolerance = hit_tolerance
        self.linear_approx_steps = linear_approx_steps
        self.skip_first = skip_first


    def set_models(
        self, *models: RadiativeModel,
    ):
        self.models = [*models]
        if len(self.models) < 1:
            raise ValueError("No Radiative models found")
        
    def compute(self, trajectory: Trajectory, history: bool = False, skip: int = None, device: str = None, dtype: str = None):
        """
        Computes the radiative transfer for a given trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory of photons.
        """
        if len(self.models) < 1: 
            RuntimeError('No RadiativeModels to compute')

        ntraj = trajectory.ntraj
        nsteps = trajectory.nsteps
        batch_shape = trajectory.X[..., 0, 0].shape
        device = device or trajectory.X.device
        dtype = dtype or trajectory.X.dtype
        self.particle = trajectory.particle

        z = RunningTensor(torch.zeros(ntraj, device=device, dtype=dtype), trace=history)

        for model in self.models:
            model.init(batch_shape=batch_shape, trace=history, device=device, dtype=dtype)
     
        # NOTE: For backward grrt (starting from observer these should be updated)
        x_prev = trajectory.X[..., -1, :].to(device=device, dtype=dtype)
        p_prev = trajectory.P[..., -1, :].to(device=device, dtype=dtype)
        s0 = self.medium.signed_distance(x_prev)
        e0 = trajectory.P[..., 0, 0].to(device=device, dtype=dtype)

        all_hits = [torch.zeros(batch_shape, device=device, dtype=torch.bool)]*2
        for i in tqdm.trange(nsteps-2, 0, -1):
            x_new = trajectory.X[..., i, :].to(device=device, dtype=dtype)
            p_new = trajectory.P[..., i, :].to(device=device, dtype=dtype)
            mask_new = trajectory._genuine_steps[..., i].to(device=device, dtype=torch.bool)
            dlambda = abs(trajectory.affine_t[i] - trajectory.affine_t[i+1]).to(device=device, dtype=dtype)
            
            s0, hits = self._step(
                mask=mask_new,
                z=z,
                x_prev=x_prev,
                x_new=x_new,
                p_prev=p_prev,
                p_new=p_new,
                dlambda=dlambda,
                e0=e0,
                s0=s0,
            )
            all_hits.append(hits.clone())
            x_prev, p_prev = x_new, p_new

        return GRRTData(hits=all_hits, z=z, fluxes=[])


    def _step(
        self,
        z: RunningTensor,
        mask: torch.Tensor,
        x_prev: torch.Tensor,
        x_new: torch.Tensor,
        p_prev: torch.Tensor,
        p_new: torch.Tensor,
        dlambda: torch.Tensor,
        e0: torch.Tensor,
        s0: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        s1 = self.medium.signed_distance(x_new)
        is_hit = self.medium.hit_condition(s0, s1)
        mask = is_hit & mask

        if not torch.any(is_hit):
            return s1, is_hit
        
        x_prev_hit = x_prev[is_hit]
        x_new_hit = x_new[is_hit]
        p_prev_hit = p_prev[is_hit]
        p_new_hit = p_new[is_hit]  
        s0_hit = s0[is_hit]
        s1_hit = s1[is_hit]

        x_hit, p_hit = self.medium.adjust_hit(
            x_prev_hit, x_new_hit, p_prev_hit, p_new_hit, s0_hit, s1_hit
        )

        z_ = self.redshift(x_hit, p_hit, e0[is_hit])
        z.update(z_, is_hit)

        for model in self.models:  
            model.update(
                mask=is_hit, x=x_hit, z=z_, dlambda=dlambda, medium=self.medium
            )

        return s1, is_hit

    def redshift(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        e0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the total redshift (gravitational + Doppler).
        z = E_source / E_obs - 1

        Parameters
        ----------

        Returns
        -------
        torch.Tensor
            the redshift
        """
        # fluid velocity u^a
        u_a = self.medium.velocity(x)
        # LRNF tetrad - always the base space metric should be used
        e_a_mu = self.medium.spacetime.lnrf(x)
        p_a = torch.einsum("...ij,...j->...i", e_a_mu, p)

        # u^a p_a, energy in comoving frame
        energy_comoving = - torch.einsum("...i,...i->...", u_a, p_a)

        # Energy of stationary observer at infinity. Assumes stationary spacetime.
        # energy_obs = - p_a[..., 0]
        energy_obs = - e0

        z = energy_comoving / energy_obs - 1.0
        return z

    def hook(self, x: torch.Tensor, p: torch.Tensor):
        """
        A hook for on-the-fly computation of the GRRT model.
        """
        # TBD
        raise NotImplementedError
