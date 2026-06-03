import torch

from bhtrace.medium import Medium
from bhtrace.grrt.radiation._base import RadiativeModel, RADIATIVE_MODEL_REGISTRY


@RADIATIVE_MODEL_REGISTRY.register("bolometric_flux")
class BolometricFlux(RadiativeModel):
    """
    A radiative model for bolometric flux.

    This model uses the medium's flux density to evaluate radiative transfer
    along the ray. It assumes that the medium is optically thin and therefore
    ignores absorption. The change in intensity is calculated by adding the
    source term, corrected for the Doppler factor, to the current intensity.

    This model is suitable for scenarios where the emission is known
    bolometrically and absorption can be neglected.
    """

    def __init__(self):
        """
        Initializes the BolometricFlux model.

        The spectral dimension is set to 1, as this model deals with
        bolometric (integrated) quantities.
        """
        super().__init__(1)

    def step(
        self,
        x: torch.Tensor,
        intensity: torch.Tensor,
        z: torch.Tensor,
        dlambda: torch.Tensor,
        medium: Medium,
    ) -> torch.Tensor:
        """
        Performs a single step of bolometric radiative transfer.

        The new intensity is computed by adding the medium's flux density,
        scaled by the Doppler factor, to the previous intensity.

        Parameters
        ----------
        x : torch.Tensor
            The position(s) at which to compute the radiation, with shape
            (..., 4).
        intensity : torch.Tensor
            The invariant bolometric intensity from the previous step, with
            shape (..., 1).
        z : torch.Tensor
            The redshift(s), with shape (..., 1).
        dlambda : torch.Tensor
            The affine parameter differential. This is not used in this model
            as it assumes an optically thin medium.
        medium : bhtrace.medium.Medium
            The medium through which the radiation propagates.

        Returns
        -------
        torch.Tensor
            The new invariant bolometric intensity, with shape (..., 1).
        """

        source = medium.flux_density(x).unsqueeze(-1)
        doppler_factor = (1 + z).pow(-4).unsqueeze(-1)

        return intensity + (source * doppler_factor)
