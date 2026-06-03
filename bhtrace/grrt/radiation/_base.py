from typing import Tuple
from abc import ABC, abstractmethod

import torch

from bhtrace.utils import Registry
from bhtrace.medium import Medium
from bhtrace.data import RunningTensor


class RadiativeModel(ABC):
    r"""
    Base class for radiation models to be used with the `GRRT` class.

    This class provides an additional level of abstraction for `GRRT`
    calculations. It uses local physical properties of the medium to compute
    local radiative properties according to a certain model and then provides a
    high-level API for the radiative transfer solver.

    Attributes
    ----------
    sdim : int
        The dimension of the spectral data (e.g., number of frequency bins).
    clip : bool
        Whether to clip the computed intensities to non-negative values.
    zeronan : bool
        Whether to replace NaN values in the computed intensities with zeros.
    intensities : bhtrace.data.RunningTensor
        A data container for the computed intensities. See
        `bhtrace.data.RunningTensor` for more details.

    Methods
    -------
    init(batch_shape, trace, device, dtype)
        Prepares the internal state for a GRRT run.
    update(mask, x, z, dlambda, medium)
        Performs one step of the radiative transfer calculation for this model.
    step(x, intensity, z, dlambda, medium)
        Computes the new intensity based on the physical properties of the
        medium at a given point.
    """
    def __init__(self, sdim: int, clip: bool = True, zeronan: bool = True):
        """
        Initializes the RadiativeModel.

        Parameters
        ----------
        sdim : int
            The dimension of the spectral data (e.g., number of frequency bins).
        clip : bool, optional
            Whether to clip the computed intensities to non-negative values.
            Default is True.
        zeronan : bool, optional
            Whether to replace NaN values in the computed intensities with
            zeros. Default is True.
        """
        self.sdim = sdim
        self.clip = clip
        self.zeronan = zeronan

    def init(self, batch_shape: Tuple[int], trace: bool, device: str, dtype: str) -> torch.Tensor:
        """Prepares internal state for a GRRT run.

        Initializes the `intensities` tensor with zeros and sets up the
        `RunningTensor` for tracking them.

        Parameters
        ----------
        batch_shape : tuple[int]
            The shape of the batch (trajectory) dimension.
        trace : bool
            A flag to indicate whether to keep a history of updates.
        device : str
            The `torch.device` on which the GRRT calculations will be done.
        dtype : str
            The `torch.dtype` in which the GRRT calculations will be done.

        Returns
        -------
        torch.Tensor
            The initial intensity tensor, filled with zeros.
        """
        x0 = torch.zeros([*batch_shape, self.sdim], device=device, dtype=dtype)
        self.intensities = RunningTensor(x0=x0, trace=trace)
        return x0

    def update(
        self,
        mask: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ):
        """
        Performs one step of the radiative transfer calculation for this model.

        This method isolates the stateful update logic for the `.intensity`
        attribute from the actual step logic.

        Parameters
        ----------
        mask : torch.Tensor[bool]
            A hit mask for the batch dimension, indicating for which particles
            to update the intensities.
        x : torch.Tensor
            The position(s) at which to compute the radiation (masked), with shape
            (..., 4).
        z : torch.Tensor
            The redshift(s) (masked), with shape (..., 1).
        dlambda : torch.Tensor
            The affine parameter differential(s) (masked), with shape (...) or
            (1,).
        medium : bhtrace.medium.Medium
            The medium through which the radiation propagates.
        """

        new_i = self.step(
            x=x, z=z, intensity=self.intensities.x[mask], dlambda=dlambda, medium=medium
        )
        if self.zeronan:
            new_i = new_i.nan_to_num()
        if self.clip:
            new_i = new_i.clip(min=0)

        self.intensities.update(new_i, mask)

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        intensity: torch.Tensor,
        z: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:
        """
        Computes the new intensity based on physical properties of the medium.

        This method must be implemented by all subclasses.

        Parameters
        ----------
        x : torch.Tensor
            The position(s) at which to compute the radiation, with shape
            (..., 4).
        intensity : torch.Tensor
            The invariant (specific) intensity on the previous step, with shape
            (..., sdim).
        z : torch.Tensor
            The redshift(s), with shape (..., 1).
        dlambda : torch.Tensor
            The affine parameter differential(s), with shape (...) or (1,).
        medium : bhtrace.medium.Medium
            The medium for which to compute the radiation.

        Returns
        -------
        torch.Tensor
            The new invariant intensity, with shape (..., sdim).
        """
        raise NotImplementedError()


RADIATIVE_MODEL_REGISTRY = Registry(RadiativeModel)
