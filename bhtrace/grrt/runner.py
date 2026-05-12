"""
This file describes a wrapper for medium models, which performs GRRT calculation


"""

from typing import Any, Optional, Dict, Literal, Tuple, List, Iterable
from dataclasses import dataclass, field

import torch
import scipy.constants as constants


from bhtrace.geometry import Spacetime, Particle
from bhtrace.medium import Medium
from bhtrace.trajectory import Trajectory
from bhtrace.grrt.radiation import RadiativeModel

@dataclass
class GRRTHistory:
    x: List[torch.Tensor] = field(default_factory=list)
    p: List[torch.Tensor] = field(default_factory=list)
    is_hit: List[torch.Tensor] = field(default_factory=list)
    z: List[torch.Tensor] = field(default_factory=list)
    dtau: List[torch.Tensor] = field(default_factory=list)

class GRRT:
    """
    Attributes
    ----------
    images : Dict[str | float, torch.Tensor]
        container for processed intensitites

    Methods
    -------

    compute()
        Computes GRRT model for already traced trajectories

    hook()
        A hook to pass as adjoint to the Tracer for computing trajectories on-the-fly

    retrieve()
        Retrieves computed grrt image.

    Notes
    -----
    Total flux along ray can be computed in assumption of optically thin disk
    """

    images: Dict[str | float, torch.Tensor]
    sky: torch.Tensor

    def __init__(
        self,
        medium: Medium,
        compute_total: bool,
        frequences: Optional[Iterable[float]] = None,
        lines: Optional[Iterable[float]] = None,
        hit_tolerance: float = 1e-3,
        linear_approx_steps: int = 1,
        skip_first: int = 1,
        probe_idx: Optional[int] = None,
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
            Number of initial steps to skip. Defaults to 1.
        """
        self.medium = medium
        self.hit_tolerance = hit_tolerance
        self.linear_approx_steps = linear_approx_steps
        self.skip_first = skip_first
        self.probe_idx = probe_idx
        self.probe_history: Optional[GRRTHistory] = None

        frequences = frequences or []
        lines = lines or []
        _frequences = [v for v in frequences] + [constants.c / l for l in lines]

        if len(_frequences) > 0:
            self.frequences = torch.tensor(_frequences)
        else:
            self.frequences = None

        self.compute_total = compute_total
        self.compute_spectrum = self.frequences is not None

        self.total_flux: Optional[torch.Tensor] = None
        self.spectrum: Optional[torch.Tensor] = None
        self.images = {}

    def attach_models(self, total_models: List[RadiativeModel] = [], spectral_models: List[RadiativeModel] = []):
        self.total_models = total_models
        self.spectral_models = spectral_models

    def compute(self, trajectory: Trajectory):
        """
        Computes the radiative transfer for a given trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory of photons.
        """
        if not self.total_models and not self.spectral_models:
            raise RuntimeError("Radiative models not attached. Call attach_models() first.")

        if self.probe_idx is not None:
            self.probe_history = GRRTHistory()

        # Initialize intensities
        if self.compute_total:
            # one value per trajectory
            self.total_flux = torch.zeros(
                trajectory.ntraj, device=trajectory.X.device, dtype=trajectory.X.dtype
            )
        if self.compute_spectrum:
            num_freqs = len(self.frequences)
            # shape: (num_trajectories, num_freqs)
            self.spectrum = torch.zeros(
                trajectory.ntraj, num_freqs, device=trajectory.X.device, dtype=trajectory.X.dtype
            )

        x_prev, p_prev = trajectory.X[..., self.skip_first, :], trajectory.P[..., self.skip_first, :]
        # print(x_prev.shape)
        self._s0 = self.medium.signed_distance(x_prev)
        e0 = trajectory.P[..., 0, 0]
        # print(e0.shape)

        for i in range(self.skip_first, len(trajectory) - 1):
            x_new, p_new = trajectory.X[..., i + 1, :], trajectory.P[..., i + 1, :]
            mask = trajectory._genuine_steps[..., i]
            dlambda = trajectory.affine_t[..., i + 1] - trajectory.affine_t[..., i]
            _, _, s1 = self._step(trajectory.particle, x_prev, x_new, p_prev, p_new, dlambda, e0, mask=mask)
            x_prev, p_prev = x_new, p_new
            self._s0 = s1

    def _step(
        self,
        particle: Particle,
        x_prev: torch.Tensor,
        x_new: torch.Tensor,
        p_prev: torch.Tensor,
        p_new: torch.Tensor,
        dlambda: torch.Tensor,
        e0: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:

        # Determine hit conditions
        s1 = self.medium.signed_distance(x_new)
        is_hit = self.medium.hit_condition(self._s0, s1)
        z = None

        # Filter out points inside the event horizon
        if hasattr(self.medium.spacetime, 'r_h'):
            r_new = x_new[..., 1] # Assumes r is the second coordinate
            is_outside_horizon = r_new > self.medium.spacetime.r_h
            is_hit = is_hit & is_outside_horizon & mask

        # If any hits, process radiative models
        if torch.any(is_hit):
            # print(is_hit.shape)
            x_prev_hit = x_prev[is_hit]
            x_new_hit = x_new[is_hit]
            p_prev_hit = p_prev[is_hit]
            p_new_hit = p_new[is_hit] # Use p_new as an approximation for p at hit point
            s0_hit = self._s0[is_hit]
            s1_hit = s1[is_hit]
            energy = e0[is_hit]
            dlambda_hit = dlambda if dlambda.numel() == 1 else dlambda[is_hit]

            x_hit, p_hit = self.medium.adjust_hit(x_prev_hit, x_new_hit, p_prev_hit, p_new_hit, s0_hit, s1_hit)

            z = self.doppler_shift(x_hit, p_hit, energy)

            if self.compute_total:
                self._flux_step(x_hit, z, is_hit)

            if self.compute_spectrum:
                nu_obs = self.frequences.to(x_hit.device, dtype=x_hit.dtype)
                nu_comoving = nu_obs.unsqueeze(0) * (1 + z).unsqueeze(-1)
                self._spectrum_step(x_hit, nu_comoving, dlambda_hit, is_hit)
        
        if self.probe_history is not None:
            probe_is_hit = is_hit[self.probe_idx]
            self.probe_history.x.append(x_new[self.probe_idx].clone())
            self.probe_history.p.append(p_new[self.probe_idx].clone())
            self.probe_history.is_hit.append(probe_is_hit)
            if probe_is_hit:
                hit_indices = is_hit.nonzero(as_tuple=True)[0]
                probe_hit_index = (hit_indices == self.probe_idx).nonzero(as_tuple=True)[0]
                self.probe_history.z.append(z[probe_hit_index].item())
            else:
                self.probe_history.z.append(0.0)

        return z, is_hit, s1


    def doppler_shift(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        e0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the total redshift (gravitational + Doppler).
        z = E_source / E_obs - 1
        """
        # fluid velocity u^a
        u_a = self.medium.velocity(x)
        # LRNF tetrad
        # print(x.shape)
        e_a_mu = self.medium.spacetime.lnrf(x)
        p_a = torch.einsum("...ij,...j->...i", e_a_mu, p)
        # p_a = e_a_mu @ p


        # u^a p_a, energy in comoving frame
        energy_comoving = - torch.einsum("...i,...i->...", u_a, p_a)

        # Energy at stationary observer at infinity. Assumes stationary spacetime.
        # E_obs = -p_t = -p_0
        # energy_obs = - p_a[..., 0]
        energy_obs = e0.abs()

        # redshift z = E_em / E_obs - 1
        z = energy_comoving / energy_obs - 1.0
        return z

    def hook(self, x: torch.Tensor, p: torch.Tensor):
        """
        A hook for on-the-fly computation of the GRRT model.
        NOTE: This hook would require the affine parameter lambda to be passed
              to properly work with all radiative models.
        """
        # TBD
        raise NotImplementedError

    def retrieve(self, kind: Literal['total', 'spectrum'] = 'spectrum') -> torch.Tensor:
        """
        Retrieves computed grrt image.

        Parameters
        ----------
        kind : Literal['total', 'spectrum'], optional
            Which result to retrieve. Defaults to 'spectrum'.

        Returns
        -------
        torch.Tensor
            The computed flux or spectrum.
        """
        if kind == 'total':
            if self.total_flux is None:
                raise RuntimeError("Total flux was not computed. Call compute() first.")
            return self.total_flux
        elif kind == 'spectrum':
            if self.spectrum is None:
                raise RuntimeError("Spectrum was not computed. Call compute() first.")
            return self.spectrum
        else:
            raise ValueError(f"Unknown kind: {kind}. Available: 'total', 'spectrum'")

    def _spectrum_step(self, x: torch.Tensor, nu_comoving: torch.Tensor, dlambda: torch.Tensor, mask: torch.Tensor):
        """
        Performs a step for the spectral calculation.
        """
        current_spectrum = self.spectrum[mask, :]
        new_spectrum = current_spectrum
        for model in self.spectral_models:
            new_spectrum = model.step(x, new_spectrum, nu_comoving, dlambda, self.medium)
        self.spectrum[mask, :] = new_spectrum


    def _flux_step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        Performs a step for the total flux calculation.
        """
        new_flux = self.total_flux[mask]
        for model in self.total_models:
            new_flux = model.step(x, new_flux, z, self.medium)
        self.total_flux[mask] = new_flux
