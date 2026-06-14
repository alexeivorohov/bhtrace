import torch

from bhtrace.medium import Medium
from bhtrace.grrt.radiation._base import RadiativeModel, RADIATIVE_MODEL_REGISTRY

@RADIATIVE_MODEL_REGISTRY.register('blackbody_absorption')
class Blackbody(RadiativeModel):
    """
    A radiative model for blackbody radiation with absorption.

    This model implements the radiative transfer equation for an optically
    thick or thin medium emitting and absorbing as a blackbody. It handles
    both the optically thin and thick regimes by using a different form of
    the equation based on the optical depth `dtau`.

    Attributes
    ----------
    frequences : torch.Tensor
        The frequencies at which the spectrum is computed (in Hz).
    dtau_thick : float
        The threshold for optical depth to switch between the optically thin
        and thick approximations.
    """
    def __init__(self, spectrum: torch.Tensor, dtau_thick=1e-2, clip: bool = True, zeronan: bool = True):
        """
        Initializes the Blackbody model.

        Parameters
        ----------
        spectrum : torch.Tensor
            A 1D tensor of frequencies to compute the spectrum for (in Hz).
        dtau_thick : float, optional
            The threshold for optical depth to switch between the optically
            thin and thick approximations. Default is 1e-2.
        clip : bool, optional
            Whether to clip the computed intensities to non-negative values.
            Default is True.
        zeronan : bool, optional
            Whether to replace NaN values in the computed intensities with
            zeros. Default is True.
        """
        self.frequences = spectrum.flatten()
        super().__init__(self.frequences.numel(), clip=clip, zeronan=zeronan)
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
        Performs a single step of blackbody radiative transfer.

        This method calculates the change in intensity due to blackbody
        emission and absorption. It distinguishes between optically thin and
        thick regimes based on the `dtau_thick` threshold.

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
            The affine parameter differential(s), with shape (...) or (1,).
        medium : bhtrace.medium.Medium
            The medium through which the radiation propagates.

        Returns
        -------
        torch.Tensor
            The new invariant specific intensity, with shape (..., sdim).
        """

        temp = medium.temperature(x).unsqueeze(-1)
        rest_mass_density = medium.rest_mass_density(x).unsqueeze(-1)
        opacity = medium.opacity(x).unsqueeze(-1)

        nu_comoving = medium._nu_scale * self.frequences.to(dtype=x.dtype, device=x.device) * (1 + z).unsqueeze(-1)
        inv_alpha = rest_mass_density * opacity * nu_comoving

        mu = (medium._bb_pow * nu_comoving / temp).nan_to_num(float('+inf'))
        radiated = medium._bb_scale * nu_comoving.pow(3) / torch.expm1(mu)

        dtau = inv_alpha * dlambda.unsqueeze(-1)
        new_inv_i = torch.zeros_like(intensity)

        thick = dtau > self.dtau_thick
        thin = ~thick

        if thin.any():
            new_inv_i[thin] = (
                intensity[thin] + (radiated[thin] - intensity[thin]) * dtau[thin]
            )

        if thick.any():
            exp_dtau_thick = (-dtau[thick]).exp()
            new_inv_i[thick] = intensity[thick] * exp_dtau_thick + radiated[thick] * (
                1.0 - exp_dtau_thick
            )

        return new_inv_i
