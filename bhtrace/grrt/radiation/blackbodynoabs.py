import torch

from bhtrace.medium import Medium
from bhtrace.grrt.radiation._base import RadiativeModel, RADIATIVE_MODEL_REGISTRY


@RADIATIVE_MODEL_REGISTRY.register("blackbody")
class BlackbodyN(RadiativeModel):
    """
    A model for blackbody radiation without absorption.

    This model is designed for test purposes. It calculates the emitted
    blackbody radiation at each step but does not account for absorption.
    The new intensity is simply the previous intensity plus the newly
    radiated amount.

    Attributes
    ----------
    frequences : torch.Tensor
        The frequencies at which the spectrum is computed (in Hz).
    dtau_thick : float
        This parameter is inherited but not used, as absorption is ignored.
    """
    def __init__(self, spectrum: torch.Tensor, dtau_thick=1e-2):
        """
        Initializes the BlackbodyN model.

        Parameters
        ----------
        spectrum : torch.Tensor
            A 1D tensor of frequencies to compute the spectrum for (in Hz).
        dtau_thick : float, optional
            This parameter is accepted but not used in this model.
            Default is 1e-2.
        """
        self.frequences = spectrum.flatten()
        super().__init__(self.frequences.numel())
        self.dtau_thick = dtau_thick

    def step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        intensity: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:
        """
        Performs a single step of blackbody radiative transfer without absorption.

        This method calculates the change in intensity due to blackbody
        emission only.

        Parameters
        ----------
        x : torch.Tensor
            The position(s) at which to compute the radiation, with shape
            (..., 4).
        z : torch.Tensor
            The redshift(s), with shape (..., 1).
        intensity : torch.Tensor
            The invariant specific intensity from the previous step, with shape
            (..., sdim).
        dlambda : torch.Tensor
            The affine parameter differential. This is not used in this model.
        medium : bhtrace.medium.Medium
            The medium through which the radiation propagates.

        Returns
        -------
        torch.Tensor
            The new invariant specific intensity, with shape (..., sdim).
        """
        temp = medium.temperature(x).unsqueeze(-1)
        nu_comoving = medium._nu_scale * self.frequences.to(dtype=x.dtype, device=x.device) * (1 + z).unsqueeze(-1)

        mu = (medium._bb_pow * nu_comoving / temp).nan_to_num(float('+inf'))
        radiated = medium._bb_scale * nu_comoving.pow(3) / torch.expm1(mu)

        return intensity + radiated
